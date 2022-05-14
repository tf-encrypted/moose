import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_rep_sigmoid_comp(self, x_array):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_sigmoid_comp():
            with bob:
                x = edsl.constant(x_array)
                x_enc = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with rep:
                y = edsl.sigmoid(x_enc)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_sigmoid_comp

    def _setup_float_exp_comp(self, x_array, dtype):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")

        @edsl.computation
        def my_sigmoid_comp():
            with bob:
                x = edsl.constant(x_array, dtype=dtype)

            with alice:
                y = edsl.sigmoid(x)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, dtype=dtype))

            return res

        return my_sigmoid_comp

    @parameterized.parameters(
        ([1, 0, 2, 3],),
        ([-1, 0, -2, -3.5],),
        ([-4.132, 0, -2, -3.5],),
    )
    def test_exp_example_execute(self, x):
        x_arg = np.array(x, dtype=np.float64)
        exp_comp = self._setup_rep_sigmoid_comp(x_arg)
        traced_exp_comp = edsl.trace(exp_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_exp_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        sigmoid = lambda x: 1. / (1. + np.exp(-x))
        np.testing.assert_almost_equal(actual_result, sigmoid(x_arg), decimal=5)

    @parameterized.parameters(
        ([1, 0, 2, 3], edsl.float64),
        ([-1, 0, -2, -3.5], edsl.float64),
        ([-4.132, 0, -2, -3.5], edsl.float32),
        ([-4.132, 0, -2, -3.5], edsl.float32),
    )
    def test_float_exp_execute(self, x, edsl_dtype):
        x_arg = np.array(x, dtype=edsl_dtype.numpy_dtype)
        exp_comp = self._setup_float_exp_comp(x_arg, edsl_dtype)
        traced_exp_comp = edsl.trace(exp_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_exp_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        sigmoid = lambda x: 1. / (1. + np.exp(-x))
        np.testing.assert_almost_equal(actual_result, sigmoid(x_arg), decimal=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
