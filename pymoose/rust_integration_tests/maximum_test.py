import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class MaximumExample(parameterized.TestCase):
    def _setup_max_comp(self, replicated=True):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        if replicated:

            @pm.computation
            def my_comp(
                x_uri: pm.Argument(placement=alice, vtype=pm.StringType()),
                y_uri: pm.Argument(placement=bob, vtype=pm.StringType()),
                z_uri: pm.Argument(placement=carole, vtype=pm.StringType()),
            ):
                with alice:
                    x = pm.load(x_uri, dtype=pm.float64)
                    x = pm.cast(x, dtype=pm.fixed(14, 23))

                with bob:
                    y = pm.load(y_uri, dtype=pm.float64)
                    y = pm.cast(y, dtype=pm.fixed(14, 23))

                with carole:
                    z = pm.load(z_uri, dtype=pm.float64)
                    z = pm.cast(z, dtype=pm.fixed(14, 23))

                with rep:
                    mx = pm.maximum([x, y, z])

                with bob:
                    mx_host = pm.cast(mx, dtype=pm.float64)
                    result = pm.save("maximum", mx_host)

                return result

        else:

            @pm.computation
            def my_comp(
                x_uri: pm.Argument(placement=alice, vtype=pm.StringType()),
                y_uri: pm.Argument(placement=bob, vtype=pm.StringType()),
                z_uri: pm.Argument(placement=carole, vtype=pm.StringType()),
            ):
                with alice:
                    x = pm.load(x_uri, dtype=pm.float64)
                    x = pm.cast(x, dtype=pm.fixed(14, 23))

                with carole:
                    z = pm.load(z_uri, dtype=pm.float64)
                    z = pm.cast(z, dtype=pm.fixed(14, 23))

                with bob:
                    y = pm.load(y_uri, dtype=pm.float64)
                    y = pm.cast(y, dtype=pm.fixed(14, 23))
                    mx = pm.cast(pm.maximum([x, y, z]), dtype=pm.float64)
                    result = pm.save("maximum", mx)

                return result

        return my_comp

    def _setup_float_max_comp(self, edsl_type):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")

        @pm.computation
        def my_comp(
            x_uri: pm.Argument(placement=bob, vtype=pm.StringType()),
            y_uri: pm.Argument(placement=bob, vtype=pm.StringType()),
            z_uri: pm.Argument(placement=carole, vtype=pm.StringType()),
        ):
            with alice:
                x = pm.load(x_uri, dtype=edsl_type)

            with bob:
                y = pm.load(y_uri, dtype=edsl_type)

            with carole:
                z = pm.load(z_uri, dtype=edsl_type)

            with bob:
                mx = pm.maximum([x, y, z])
                result = pm.save("maximum", mx)

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
            computation=comp,
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
            pm.float64,
        ),
        (
            np.array([1.0, 2.0, 9.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([321.0, 5.0, 6.0]),
            pm.float32,
        ),
    )
    def test_float_maximum_execute(self, x, y, z, edsl_dtype):
        x_arg = np.array(x, dtype=edsl_dtype.numpy_dtype)
        y_arg = np.array(y, dtype=edsl_dtype.numpy_dtype)
        z_arg = np.array(z, dtype=edsl_dtype.numpy_dtype)

        comp = self._setup_float_max_comp(edsl_dtype)
        storage = {
            "alice": {"x_arg": x_arg},
            "bob": {"y_arg": y_arg},
            "carole": {"z_arg": z_arg},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=comp,
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
