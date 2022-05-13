import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import types as ty
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class MaximumExample(parameterized.TestCase):
    def _setup_max_comp(self, replicated=True):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        if replicated:

            @edsl.computation
            def my_comp(
                x_uri: edsl.Argument(placement=alice, vtype=ty.StringType()),
                y_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
                z_uri: edsl.Argument(placement=carole, vtype=ty.StringType()),
            ):
                with alice:
                    x = edsl.load(x_uri, dtype=edsl.float64)
                    x = edsl.cast(x, dtype=edsl.fixed(14, 23))

                with bob:
                    y = edsl.load(y_uri, dtype=edsl.float64)
                    y = edsl.cast(y, dtype=edsl.fixed(14, 23))

                with carole:
                    z = edsl.load(z_uri, dtype=edsl.float64)
                    z = edsl.cast(z, dtype=edsl.fixed(14, 23))

                with rep:
                    mx = edsl.maximum([x, y, z])

                with bob:
                    mx_host = edsl.cast(mx, dtype=edsl.float64)
                    result = edsl.save("maximum", mx_host)

                return result

        else:

            @edsl.computation
            def my_comp(
                x_uri: edsl.Argument(placement=alice, vtype=ty.StringType()),
                y_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
                z_uri: edsl.Argument(placement=carole, vtype=ty.StringType()),
            ):
                with alice:
                    x = edsl.load(x_uri, dtype=edsl.float64)
                    x = edsl.cast(x, dtype=edsl.fixed(14, 23))

                with carole:
                    z = edsl.load(z_uri, dtype=edsl.float64)
                    z = edsl.cast(z, dtype=edsl.fixed(14, 23))

                with bob:
                    y = edsl.load(y_uri, dtype=edsl.float64)
                    y = edsl.cast(y, dtype=edsl.fixed(14, 23))
                    mx = edsl.cast(edsl.maximum([x, y, z]), dtype=edsl.float64)
                    result = edsl.save("maximum", mx)

                return result

        return my_comp

    def _setup_float_max_comp(self, edsl_type):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")

        @edsl.computation
        def my_comp(
            x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
            y_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
            z_uri: edsl.Argument(placement=carole, vtype=ty.StringType()),
        ):
            with alice:
                x = edsl.load(x_uri, dtype=edsl_type)

            with bob:
                y = edsl.load(y_uri, dtype=edsl_type)

            with carole:
                z = edsl.load(z_uri, dtype=edsl_type)

            with bob:
                mx = edsl.maximum([x, y, z])
                result = edsl.save("maximum", mx)

            return result

        return my_comp

    @parameterized.parameters(
        (
            np.array([1.0, 2.0, 9.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            True,
        ),
        (
            np.array([1.0, 2.0, 9.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            False,
        ),
    )
    def test_maximum_fixed(self, x, y, z, run_rep):
        comp = self._setup_max_comp(replicated=run_rep)
        traced_maximum_comp = edsl.trace(comp)

        x_arg = np.array(x, dtype=np.float64)
        y_arg = np.array(y, dtype=np.float64)
        z_arg = np.array(z, dtype=np.float64)

        storage_rep = {
            "alice": {"x_arg": x_arg},
            "bob": {"y_arg": y_arg},
            "carole": {"z_arg": z_arg},
        }

        runtime_rep = LocalMooseRuntime(storage_mapping=storage_rep)
        _ = runtime_rep.evaluate_computation(
            computation=traced_maximum_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg", "y_uri": "y_arg", "z_uri": "z_arg"},
        )

        result = runtime_rep.read_value_from_storage("bob", "maximum")

        np.testing.assert_equal(result, np.maximum(x_arg, np.maximum(y_arg, z_arg)))

    @parameterized.parameters(
        (
            np.array([1.0, 2.0, 9.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            edsl.float64,
        ),
        (
            np.array([1.0, 2.0, 9.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([321.0, 5.0, 6.0]),
            edsl.float32,
        ),
    )
    def test_float_maximum_execute(self, x, y, z, edsl_dtype):
        x_arg = np.array(x, dtype=edsl_dtype.numpy_dtype)
        y_arg = np.array(y, dtype=edsl_dtype.numpy_dtype)
        z_arg = np.array(z, dtype=edsl_dtype.numpy_dtype)

        comp = self._setup_float_max_comp(edsl_dtype)
        traced_maximum_comp = edsl.trace(comp)
        storage = {
            "alice": {"x_arg": x_arg},
            "bob": {"y_arg": y_arg},
            "carole": {"z_arg": z_arg},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_maximum_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg", "y_uri": "y_arg", "z_uri": "z_arg"},
        )

        actual_result = runtime.read_value_from_storage("bob", "maximum")
        np.testing.assert_equal(
            actual_result, np.maximum(x_arg, np.maximum(y_arg, z_arg))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="concat example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
