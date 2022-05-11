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
    def _setup_comp(self, replicated=True):
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
                x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
                y_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
                z_uri: edsl.Argument(placement=carole, vtype=ty.StringType()),
            ):
                with alice:
                    x = edsl.load(x_uri, dtype=edsl.float64)

                with bob:
                    y = edsl.load(y_uri, dtype=edsl.float64)

                with carole:
                    z = edsl.load(z_uri, dtype=edsl.float64)

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
    def test_maximum(self, x, y, z, run_rep):
        comp = self._setup_comp(replicated=run_rep)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="concat example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
