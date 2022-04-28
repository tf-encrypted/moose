import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation import types as ty
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class BooleanLogicExample(parameterized.TestCase):
    def _setup_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp(
            x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
            y_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
            ya_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
        ):
            with bob:
                x = edsl.load(x_uri, dtype=edsl.float64)
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

                y = edsl.load(y_uri, dtype=edsl.float64)
                y = edsl.cast(y, dtype=edsl.fixed(8, 27))

            with rep:
                z_less = edsl.less(x, y)
                z_greater = edsl.greater(x, y)
                z_twice = edsl.concatenate([z_less, z_less])
                z_mux = edsl.mux(z_less, x, y)

            with alice:
                zl_alice = edsl.logical_or(z_less, z_less)
                zg_alice = edsl.logical_or(z_greater, z_greater)
                zm_alice = edsl.cast(z_mux, dtype=edsl.float64)

                y_alice = edsl.load(ya_uri, dtype=edsl.float64)
                r_alice = (
                    edsl.save("z0", edsl.index_axis(zl_alice, axis=0, index=0)),
                    edsl.save("z1", edsl.index_axis(zl_alice, axis=0, index=1)),
                    edsl.save("z2", edsl.index_axis(zl_alice, axis=0, index=2)),
                    edsl.save("less_result", zl_alice),
                    edsl.save("greater_result", zg_alice),
                    edsl.save("y0", edsl.index_axis(y_alice, axis=0, index=2)),
                    edsl.save("mux", zm_alice),
                    edsl.save("z_twice", z_twice),
                )

            return r_alice

        return my_comp

    @parameterized.parameters(
        ([1.5, 2.3, 3, 3], [-1.0, 4.0, 3, 2]),
    )
    def test_bool_example_execute(self, x, y):
        less_comp = self._setup_comp()
        traced_less_comp = edsl.trace(less_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        x = np.array(x)
        y = np.array(y)
        z = x < y

        storage = {
            "alice": {"ya_arg": y},
            "carole": {},
            "bob": {"x_arg": x, "y_arg": y},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg", "y_uri": "y_arg", "ya_uri": "ya_arg"},
        )

        z0 = runtime.read_value_from_storage("alice", "z0")
        z1 = runtime.read_value_from_storage("alice", "z1")
        z2 = runtime.read_value_from_storage("alice", "z2")

        # testing index axis
        np.testing.assert_equal(z0, z[0])
        np.testing.assert_equal(z1, z[1])
        np.testing.assert_equal(z2, z[2])
        np.testing.assert_equal(runtime.read_value_from_storage("alice", "y0"), 3)

        # test comparison
        np.testing.assert_equal(
            runtime.read_value_from_storage("alice", "less_result"), x < y
        )

        # test greater
        np.testing.assert_equal(
            runtime.read_value_from_storage("alice", "greater_result"), x > y
        )

        # test mux
        np.testing.assert_almost_equal(
            runtime.read_value_from_storage("alice", "mux"),
            (x < y) * x + (1 - (x < y)) * y,
        )

        # test concat
        np.testing.assert_almost_equal(
            runtime.read_value_from_storage("alice", "z_twice"), np.concatenate([z, z])
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
