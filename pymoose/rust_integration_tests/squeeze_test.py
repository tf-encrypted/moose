import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation import types as ty
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class SqueezeExample(parameterized.TestCase):
    def _setup_squeeze_comp(self, axis, replicated=True):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        if replicated:

            @edsl.computation
            def my_comp(
                x_uri: edsl.Argument(placement=alice, vtype=ty.StringType()),
            ):
                with alice:
                    x = edsl.load(x_uri, dtype=edsl.float64)
                    x = edsl.cast(x, dtype=edsl.fixed(14, 23))

                with rep:
                    sq = edsl.squeeze(x, axis)

                with bob:
                    sq_host = edsl.cast(sq, dtype=edsl.float64)
                    result = edsl.save("squeeze", sq_host)

                return result

        else:

            @edsl.computation
            def my_comp(
                x_uri: edsl.Argument(placement=alice, vtype=ty.StringType()),
            ):
                with alice:
                    x = edsl.load(x_uri, dtype=edsl.float64)

                with bob:
                    sq = edsl.cast(edsl.squeeze(x, axis), dtype=edsl.float64)
                    result = edsl.save("squeeze", sq)

                return result

        return my_comp

    def _setup_float_squeeze_comp(self, axis, edsl_type):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")

        @edsl.computation
        def my_comp(
            x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
        ):
            with alice:
                x = edsl.load(x_uri, dtype=edsl_type)

            with bob:
                sq = edsl.squeeze(x, axis)
                result = edsl.save("squeeze", sq)

            return result

        return my_comp

    @parameterized.parameters(
        # (
        #     np.array([1.0, 2.0, 9.0]),
        #     np.array([1.0, 2.0, 3.0]),
        #     np.array([4.0, 5.0, 6.0]),
        #     True,
        # ),
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
        traced_squeeze_comp = edsl.trace(comp)

        storage_rep = {
            "alice": {"x_arg": x},
            "bob": {},
            "carole": {},
        }

        runtime_rep = LocalMooseRuntime(storage_mapping=storage_rep)
        _ = runtime_rep.evaluate_computation(
            computation=traced_squeeze_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg"},
        )

        result = runtime_rep.read_value_from_storage("bob", "squeeze")

        np.testing.assert_equal(result, np.squeeze(x, axis))

    @parameterized.parameters(
        (
            np.array([1.0, 2.0, 9.0]),
            None,
            edsl.float64,
        ),
        (
            np.array([[[1.0, 2.0, 9.0]]]),
            1,
            edsl.float32,
        ),
        (
            np.array([[False, True, False]]),
            0,
            edsl.bool_,
        ),
    )
    def test_float_squeeze_execute(self, x, axis, edsl_dtype):
        x_arg = np.array(x, dtype=edsl_dtype.numpy_dtype)

        comp = self._setup_float_squeeze_comp(axis, edsl_dtype)
        traced_maximum_comp = edsl.trace(comp)
        storage = {
            "alice": {"x_arg": x_arg},
            "bob": {},
            "carole": {},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_maximum_comp,
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
