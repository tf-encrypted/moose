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


class XGBoostReplicatedExample(parameterized.TestCase):
    def _setup_model_comp(self, model):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        def edsl_constant(x):
            with bob:
                return edsl.cast(
                    edsl.constant(x, dtype=edsl.float64), dtype=edsl.fixed(8, 27)
                )

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
                        features[split_indices[node]],
                        edsl_constant(split_conditions[node]),
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
                    # return edsl.constant(base_weights[node])
                    return edsl_constant(base_weights[node])

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
            prediction = edsl.add(
                edsl.add(results[0], results[1]), edsl_constant(base_score)
            )
            return prediction

        @edsl.computation
        def my_model_comp(x: edsl.Argument(bob, vtype=TensorType(edsl.float64))):
            with bob:
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with rep:
                y = map_forest(model, x)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_model_comp

    def test_xgboost_regression_example_execute(self):
        input_x = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float64)

        with  open("./pymoose/examples/xgboost/xgboost_regression_2_trees.json") as f:
            model_json = json.load(f)

        model_comp = self._setup_model_comp(model_json)
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
        expected_result = np.array([26.6495275, 88.423399], dtype=np.float64)
        np.testing.assert_almost_equal(actual_result, expected_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
