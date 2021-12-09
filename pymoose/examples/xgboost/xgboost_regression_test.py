import argparse
import json
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import base
from pymoose.computation import utils
from pymoose.computation.standard import TensorType
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


def map_tree(tree, nb_features, learning_rate, x):
    base_weights = tree["base_weights"]
    left = tree["left_children"]
    right = tree["right_children"]
    split_conditions = tree["split_conditions"]
    split_indices = tree["split_indices"]
    # Adjust weights based on learning rate
    base_weights = [learning_rate * w for w in base_weights]

    features = [edsl.index_axis(x, axis=1, index=i) for i in range(nb_features)]

    def create_computation(node):
        left_child = left[node]
        right_child = right[node]
        if left_child != -1 and right_child != -1:
            # we're at an inner node; this is the recursive case
            selector = edsl.less(
                features[split_indices[node]], edsl.constant(split_conditions[node])
            )
            return edsl.mux(
                selector,
                create_computation(left_child),
                create_computation(right_child),
            )
        else:
            assert left_child == -1
            assert right_child == -1
            # we're at a left node; this is the base case
            return edsl.constant(base_weights[node])

    return create_computation(0)


def map_forest(model, x):
    base_score = float(model["learner"]["learner_model_param"]["base_score"])
    nb_features = int(model["learner"]["learner_model_param"]["num_feature"])
    learning_rate = json.loads(model["learner"]["attributes"]["scikit_learn"])[
        "learning_rate"
    ]

    results = [
        map_tree(tree, nb_features, learning_rate, x)
        for tree in model["learner"]["gradient_booster"]["model"]["trees"]
    ]

    # TODO: to be replaced by edsl.add(edsl.addn(*results), edsl.constant(base_score))
    prediction = edsl.add(edsl.add(results[0], results[1]), edsl.constant(base_score))
    return prediction


def map_tree_hack(tree, base_weights, split_conditions, nb_features, x):
    left = tree["left_children"]
    right = tree["right_children"]
    split_indices = tree["split_indices"]

    features = [edsl.index_axis(x, axis=1, index=i) for i in range(nb_features)]

    def create_computation(node):
        left_child = left[node]
        right_child = right[node]
        if left_child != -1 and right_child != -1:
            # we're at an inner node; this is the recursive case
            selector = edsl.less(features[split_indices[node]], split_conditions[node])
            return edsl.mux(
                selector,
                create_computation(left_child),
                create_computation(right_child),
            )
        else:
            assert left_child == -1
            assert right_child == -1
            # we're at a left node; this is the base case
            return base_weights[node]

    return create_computation(0)


def extract_tree_info_hack(model, num_tree=0):
    nb_features = int(model["learner"]["learner_model_param"]["num_feature"])
    learning_rate = json.loads(model["learner"]["attributes"]["scikit_learn"])[
        "learning_rate"
    ]
    tree = model["learner"]["gradient_booster"]["model"]["trees"][num_tree]
    split_conditions = tree["split_conditions"]
    # Adjust weights based on learning rate
    base_weights = [learning_rate * w for w in tree["base_weights"]]
    return tree, nb_features, split_conditions, base_weights


class XGBoostReplicatedExample(parameterized.TestCase):
    def _setup_model_comp(self, model):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_model_comp(x: edsl.Argument(bob, vtype=TensorType(edsl.float64)),):
            with bob:
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with rep:
                y = map_forest(model, x)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_model_comp

    def _setup_single_tree_hack_comp(self, model):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_model_comp(x: edsl.Argument(bob, vtype=TensorType(edsl.float64)),):
            tree, nb_features, split_conditions, base_weights = extract_tree_info_hack(
                model
            )

            with bob:
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))
                split_conditions = [
                    edsl.cast(
                        edsl.constant(c, dtype=edsl.float64), dtype=edsl.fixed(8, 27)
                    )
                    for c in split_conditions
                ]
                base_weights = [
                    edsl.cast(
                        edsl.constant(w, dtype=edsl.float64), dtype=edsl.fixed(8, 27)
                    )
                    for w in base_weights
                ]

            with rep:
                y = map_tree_hack(tree, base_weights, split_conditions, nb_features, x)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_model_comp

    def test_single_decision_tree_hack_example_execute(self):
        input_x = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float64)
        json_file = open("./pymoose/examples/xgboost/xgboost_regression_2_trees.json")
        model_json = json.load(json_file)
        model_comp = self._setup_single_tree_hack_comp(model_json)
        traced_model_comp = edsl.trace(model_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_model_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x": input_x},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        expected_result = np.array([-42.5642625, 96.762], dtype=np.float64)

        np.testing.assert_almost_equal(actual_result, expected_result)

    # def test_xgboost_regression_example_serde(self):
    #     json_file = open("./pymoose/examples/xgboost/xgboost_regression_2_trees.json")
    #     model_json = json.load(json_file)
    #     model_comp = self._setup_model_comp(model_json)
    #     traced_model_comp = edsl.trace(model_comp)
    #     comp_bin = utils.serialize_computation(traced_model_comp)
    #     # Compile in Rust
    #     # If this does not error, rust was able to deserialize the pycomputation
    #     elk_compiler.compile_computation(comp_bin, [])

    # def test_xgboost_regression_example_compile(self):
    #     json_file = open("./pymoose/examples/xgboost/xgboost_regression_2_trees.json")
    #     model_json = json.load(json_file)
    #     model_comp = self._setup_model_comp(model_json)
    #     traced_model_comp = edsl.trace(model_comp)
    #     comp_bin = utils.serialize_computation(traced_model_comp)
    #     _ = elk_compiler.compile_computation(
    #         comp_bin,
    #         [
    #             "typing",
    #             "full",
    #             "toposort",
    #             # "print",
    #         ],
    #     )

    # def test_xgboost_regression_example_execute(self):
    #     input_x = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float64)
    #     json_file = open("./pymoose/examples/xgboost/xgboost_regression_2_trees.json")
    #     model_json = json.load(json_file)
    #     model_comp = self._setup_model_comp(model_json)
    #     traced_model_comp = edsl.trace(model_comp)
    #     storage = {
    #         "alice": {},
    #         "bob": {},
    #         "carole": {},
    #     }
    #     runtime = LocalMooseRuntime(storage_mapping=storage)
    #     _ = runtime.evaluate_computation(
    #         computation=traced_model_comp,
    #         role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
    #         arguments={"x": input_x},
    #     )
    #     actual_result = runtime.read_value_from_storage("alice", "y_uri")
    #     expected_result = np.array([26.6495275 88.423399], dtype=np.float64)

    #     np.testing.assert_almost_equal(actual_result, expected_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
