import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose import runtime as rt
from pymoose.logger import get_logger


class TransposeExample(parameterized.TestCase):
    def _setup_transpose_comp(self, replicated=True):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        if replicated:

            @pm.computation
            def my_comp(
                x_uri: pm.Argument(placement=alice, vtype=pm.StringType()),
            ):
                with alice:
                    x = pm.load(x_uri, dtype=pm.float64)
                    x = pm.cast(x, dtype=pm.fixed(14, 23))

                with rep:
                    tx = pm.transpose(x)

                with bob:
                    tx_host = pm.cast(tx, dtype=pm.float64)
                    result = pm.save("transpose", tx_host)

                return result

        else:

            @pm.computation
            def my_comp(
                x_uri: pm.Argument(placement=alice, vtype=pm.StringType()),
            ):
                with alice:
                    x = pm.load(x_uri, dtype=pm.float64)

                with bob:
                    tx = pm.cast(pm.transpose(x), dtype=pm.float64)
                    result = pm.save("transpose", tx)

                return result

        return my_comp

    def _setup_float_transpose_comp(self, edsl_type):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")

        @pm.computation
        def my_comp(
            x_uri: pm.Argument(placement=bob, vtype=pm.StringType()),
        ):
            with alice:
                x = pm.load(x_uri, dtype=edsl_type)

            with bob:
                tx = pm.transpose(x)
                result = pm.save("transpose", tx)

            return result

        return my_comp

    @parameterized.parameters(
        # test replicated
        (
            np.array([1.0, 2.0, 9.0]),
            True,
        ),
        (
            np.zeros((2, 3, 5)),
            True,
        ),
        (
            np.zeros((1, 3, 1)),
            True,
        ),
        # test on host
        (
            np.array([1.0, 2.0, 9.0]),
            False,
        ),
        (
            np.zeros((1, 3, 1)),
            False,
        ),
        (
            np.zeros((2, 3, 5)),
            False,
        ),
    )
    def test_transpose_fixed(self, x, run_rep):
        transpose_comp = self._setup_transpose_comp(replicated=run_rep)
        storage = {
            "alice": {"x_arg": x},
        }
        runtime = rt.LocalMooseRuntime(
            ["alice", "bob", "carole"], storage_mapping=storage
        )
        _ = runtime.evaluate_computation(
            computation=transpose_comp,
            arguments={"x_uri": "x_arg"},
        )

        result = runtime.read_value_from_storage("bob", "transpose")
        np.testing.assert_equal(result, np.transpose(x))

    @parameterized.parameters(
        (
            np.array([1.0, 2.0, 9.0]),
            pm.float64,
        ),
        (
            np.array([[[1.0, 2.0, 9.0]]]),
            pm.float32,
        ),
        (
            np.array([[False, True, False]]),
            pm.bool_,
        ),
        (
            np.array([[True, True, True], [False, False, False]]),
            pm.bool_,
        ),
    )
    def test_float_transpose_execute(self, x, edsl_dtype):
        x_arg = np.array(x, dtype=edsl_dtype.numpy_dtype)

        transpose_comp = self._setup_float_transpose_comp(edsl_dtype)
        storage = {
            "alice": {"x_arg": x_arg},
        }
        runtime = rt.LocalMooseRuntime(
            ["alice", "bob", "carole"], storage_mapping=storage
        )
        _ = runtime.evaluate_computation(
            computation=transpose_comp,
            arguments={"x_uri": "x_arg"},
        )

        actual_result = runtime.read_value_from_storage("bob", "transpose")
        np.testing.assert_equal(actual_result, np.transpose(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transpose example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
