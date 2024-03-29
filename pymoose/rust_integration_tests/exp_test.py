import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose import runtime as rt
from pymoose.logger import get_logger


class ReplicatedExample(parameterized.TestCase):
    def _setup_fixed_exp_comp(self, x_array):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_exp_comp():
            with bob:
                x = pm.constant(x_array)
                x_enc = pm.cast(x, dtype=pm.fixed(8, 27))

            with rep:
                y = pm.exp(x_enc)

            with alice:
                res = pm.save("y_uri", pm.cast(y, pm.float64))

            return res

        return my_exp_comp

    def _setup_float_exp_comp(self, x_array, dtype):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")

        @pm.computation
        def my_exp_comp():
            with bob:
                x = pm.constant(x_array, dtype=dtype)

            with alice:
                y = pm.exp(x)

            with alice:
                res = pm.save("y_uri", pm.cast(y, dtype=dtype))

            return res

        return my_exp_comp

    @parameterized.parameters(
        ([1, 0, 2, 3],),
        ([-1, 0, -2, -3.5],),
        ([-4.132, 0, -2, -3.5],),
    )
    def test_exp_example_execute(self, x):
        x_arg = np.array(x, dtype=np.float64)
        exp_comp = self._setup_fixed_exp_comp(x_arg)
        runtime = rt.LocalMooseRuntime(["alice", "bob", "carole"])
        _ = runtime.evaluate_computation(
            computation=exp_comp,
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        np.testing.assert_almost_equal(actual_result, np.exp(x_arg), decimal=5)

    @parameterized.parameters(
        ([1, 0, 2, 3], pm.float64),
        ([-1, 0, -2, -3.5], pm.float64),
        ([-4.132, 0, -2, -3.5], pm.float32),
        ([-4.132, 0, -2, -3.5], pm.float32),
    )
    def test_float_exp_execute(self, x, moose_dtype):
        x_arg = np.array(x, dtype=moose_dtype.numpy_dtype)
        exp_comp = self._setup_float_exp_comp(x_arg, moose_dtype)
        runtime = rt.LocalMooseRuntime(["alice", "bob", "carole"])
        _ = runtime.evaluate_computation(
            computation=exp_comp,
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        # we can't put assert_equal exactly due to IEEE precision loss when serializing
        np.testing.assert_almost_equal(actual_result, np.exp(x_arg), decimal=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
