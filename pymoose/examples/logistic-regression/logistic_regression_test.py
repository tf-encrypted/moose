import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_model_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_model_comp(
            x: edsl.Argument(bob, vtype=edsl.TensorType(edsl.float64)),
            w: edsl.Argument(bob, vtype=edsl.TensorType(edsl.float64)),
        ):
            with bob:
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))
                w = edsl.cast(w, dtype=edsl.fixed(8, 27))

            with rep:
                y = edsl.sigmoid(edsl.dot(x, w))

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_model_comp

    def test_logistic_regression_example_serde(self):
        model_comp = self._setup_model_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    def test_logistic_regression_example_compile(self):
        model_comp = self._setup_model_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        _ = elk_compiler.compile_computation(comp_bin)

    def test_logistic_regression_example_execute(self):
        input_x = np.array([2.0, 1.0], dtype=np.float64)
        input_weights = np.array([0.5, 0.1], dtype=np.float64)
        model_comp = self._setup_model_comp()
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
            arguments={"x": input_x, "w": input_weights},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")

        def logistic_regression(input, weights):
            y = np.dot(input, weights)
            sigmoid_out = 1 / (1 + np.exp(-y))
            return sigmoid_out

        np.testing.assert_almost_equal(
            actual_result, logistic_regression(input_x, input_weights)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
