import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger


class SqueezeExample(parameterized.TestCase):
    def _setup_squeeze_comp(self, axis, replicated=True):
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
                    sq = pm.squeeze(x, axis)

                with bob:
                    sq_host = pm.cast(sq, dtype=pm.float64)
                    result = pm.save("squeeze", sq_host)

                return result

        else:

            @pm.computation
            def my_comp(
                x_uri: pm.Argument(placement=alice, vtype=pm.StringType()),
            ):
                with alice:
                    x = pm.load(x_uri, dtype=pm.float64)

                with bob:
                    sq = pm.cast(pm.squeeze(x, axis), dtype=pm.float64)
                    result = pm.save("squeeze", sq)

                return result

        return my_comp

    def _setup_float_squeeze_comp(self, axis, edsl_type):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")

        @pm.computation
        def my_comp(
            x_uri: pm.Argument(placement=bob, vtype=pm.StringType()),
        ):
            with alice:
                x = pm.load(x_uri, dtype=edsl_type)

            with bob:
                sq = pm.squeeze(x, axis)
                result = pm.save("squeeze", sq)

            return result

        return my_comp

    @parameterized.parameters(
        # test replicated
        (
            np.array([1.0, 2.0, 9.0]),
            None,
            True,
        ),
        (
            np.zeros((1, 3, 1)),
            0,
            True,
        ),
        (
            np.zeros((1, 3, 1)),
            2,
            True,
        ),
        # test on host
        (
            np.array([1.0, 2.0, 9.0]),
            None,
            False,
        ),
        (
            np.zeros((1, 3, 1)),
            0,
            False,
        ),
        (
            np.zeros((1, 3, 1)),
            2,
            False,
        ),
    )
    def test_squeeze_fixed(self, x, axis, run_rep):
        comp = self._setup_squeeze_comp(axis, replicated=run_rep)

        storage_rep = {
            "alice": {"x_arg": x},
            "bob": {},
            "carole": {},
        }

        runtime_rep = pm.LocalMooseRuntime(storage_mapping=storage_rep)
        _ = runtime_rep.evaluate_computation(
            computation=comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg"},
        )

        result = runtime_rep.read_value_from_storage("bob", "squeeze")

        np.testing.assert_equal(result, np.squeeze(x, axis))

    @parameterized.parameters(
        (
            np.array([1.0, 2.0, 9.0]),
            None,
            pm.float64,
        ),
        (
            np.array([[[1.0, 2.0, 9.0]]]),
            1,
            pm.float32,
        ),
        (
            np.array([[False, True, False]]),
            0,
            pm.bool_,
        ),
    )
    def test_float_squeeze_execute(self, x, axis, edsl_dtype):
        x_arg = np.array(x, dtype=edsl_dtype.numpy_dtype)

        comp = self._setup_float_squeeze_comp(axis, edsl_dtype)
        storage = {
            "alice": {"x_arg": x_arg},
            "bob": {},
            "carole": {},
        }

        runtime = pm.LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg"},
        )

        actual_result = runtime.read_value_from_storage("bob", "squeeze")
        np.testing.assert_equal(actual_result, np.squeeze(x, axis))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="concat example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
