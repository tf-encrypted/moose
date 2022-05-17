import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.computation import types as ty
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class BooleanLogicExample(parameterized.TestCase):
    def _setup_comp(self):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_comp(
            x_uri: pm.Argument(placement=bob, vtype=ty.StringType()),
            y_uri: pm.Argument(placement=bob, vtype=ty.StringType()),
            ya_uri: pm.Argument(placement=bob, vtype=ty.StringType()),
        ):
            with bob:
                x = pm.load(x_uri, dtype=pm.float64)
                x = pm.cast(x, dtype=pm.fixed(8, 27))

                y = pm.load(y_uri, dtype=pm.float64)
                y = pm.cast(y, dtype=pm.fixed(8, 27))

            with rep:
                z_less = pm.less(x, y)
                z_greater = pm.greater(x, y)
                z_twice = pm.concatenate([z_less, z_less])
                z_mux = pm.mux(z_less, x, y)

            with alice:
                zl_alice = pm.logical_or(z_less, z_less)
                zg_alice = pm.logical_or(z_greater, z_greater)
                zm_alice = pm.cast(z_mux, dtype=pm.float64)

                y_alice = pm.load(ya_uri, dtype=pm.float64)
                r_alice = (
                    pm.save("z0", pm.index_axis(zl_alice, axis=0, index=0)),
                    pm.save("z1", pm.index_axis(zl_alice, axis=0, index=1)),
                    pm.save("z2", pm.index_axis(zl_alice, axis=0, index=2)),
                    pm.save("less_result", zl_alice),
                    pm.save("greater_result", zg_alice),
                    pm.save("y0", pm.index_axis(y_alice, axis=0, index=2)),
                    pm.save("mux", zm_alice),
                    pm.save("z_twice", z_twice),
                )

            return r_alice

        return my_comp

    @parameterized.parameters(
        ([1.5, 2.3, 3, 3], [-1.0, 4.0, 3, 2]),
    )
    def test_bool_example_execute(self, x, y):
        less_comp = self._setup_comp()
        traced_less_comp = pm.trace(less_comp)
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
    parser = argparse.ArgumentParser(description="Boolean example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
