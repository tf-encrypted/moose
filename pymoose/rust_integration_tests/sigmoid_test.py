import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger


class ReplicatedExample(parameterized.TestCase):
    def _setup_rep_sigmoid_comp(self, x_array):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_sigmoid_comp():
            with bob:
                x = pm.constant(x_array)
                x_enc = pm.cast(x, dtype=pm.fixed(8, 27))

            with rep:
                y = pm.sigmoid(x_enc)

            with alice:
                res = pm.save("y_uri", pm.cast(y, pm.float64))

            return res

        return my_sigmoid_comp

    def _setup_float_sigmoid_comp(self, x_array, dtype):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")

        @pm.computation
        def my_sigmoid_comp():
            with bob:
                x = pm.constant(x_array, dtype=dtype)

            with alice:
                y = pm.sigmoid(x)

            with alice:
                res = pm.save("y_uri", pm.cast(y, dtype=dtype))

            return res

        return my_sigmoid_comp

    @parameterized.parameters(
        ([1, 0, 2, 3],),
        ([-1, 0, -2, -3.5],),
        ([-4.132, 0, -2, -3.5],),
    )
    def test_sigmoid_example_execute(self, x):
        x_arg = np.array(x, dtype=np.float64)
        sigmoid_comp = self._setup_rep_sigmoid_comp(x_arg)
        traced_sigmoid_comp = pm.trace(sigmoid_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = pm.LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_sigmoid_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        np.testing.assert_almost_equal(actual_result, sigmoid(x_arg), decimal=5)

    @parameterized.parameters(
        ([1, 0, 2, 3], pm.float64),
        ([-1, 0, -2, -3.5], pm.float64),
        ([-4.132, 0, -2, -3.5], pm.float32),
        ([-4.132, 0, -2, -3.5], pm.float32),
    )
    def test_float_sigmoid_execute(self, x, dtype):
        x_arg = np.array(x, dtype=dtype.numpy_dtype)
        sigmoid_comp = self._setup_float_sigmoid_comp(x_arg, dtype)
        traced_sigmoid_comp = pm.trace(sigmoid_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = pm.LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_sigmoid_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        np.testing.assert_almost_equal(actual_result, sigmoid(x_arg), decimal=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
