import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger


class ReplicatedExample(parameterized.TestCase):
    def _setup_model_comp(self):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_model_comp(
            x: pm.Argument(bob, vtype=pm.TensorType(pm.float64)),
            w: pm.Argument(bob, vtype=pm.TensorType(pm.float64)),
        ):
            with bob:
                x = pm.cast(x, dtype=pm.fixed(8, 27))
                w = pm.cast(w, dtype=pm.fixed(8, 27))

            with rep:
                y = pm.sigmoid(pm.dot(x, w))

            with alice:
                res = pm.save("y_uri", pm.cast(y, pm.float64))

            return res

        return my_model_comp

    def test_logistic_regression_example_execute(self):
        input_x = np.array([2.0, 1.0], dtype=np.float64)
        input_weights = np.array([0.5, 0.1], dtype=np.float64)
        model_comp = self._setup_model_comp()
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = pm.LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=model_comp,
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
