import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class BooleanLogicExample(parameterized.TestCase):
    def _setup_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp():
            with bob:
                x = edsl.constant(np.array([1.5, 2.3, 3, 3], dtype=np.float64))
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

                y = edsl.constant(np.array([-1.0, 4.0, 3, 2], dtype=np.float64))
                y = edsl.cast(y, dtype=edsl.fixed(8, 27))

            with rep:
                z_rep = edsl.less(x, y)

            with alice:
                z_alice = edsl.logical_or(z_rep, z_rep)
                y_alice = edsl.constant(np.array([-1.0, 4.0, 3, 2], dtype=np.float64))

                r_alice = (
                    edsl.save("z0", edsl.index_axis(z_alice, axis=0, index=0)),
                    edsl.save("z1", edsl.index_axis(z_alice, axis=0, index=1)),
                    edsl.save("z2", edsl.index_axis(z_alice, axis=0, index=2)),
                    edsl.save("less_result", z_alice),
                    edsl.save("y0", edsl.index_axis(y_alice, axis=0, index=2)),
                )

            return r_alice

        return my_comp

    def test_bool_example_execute(self):
        less_comp = self._setup_comp()
        traced_less_comp = edsl.trace(less_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        z = np.array([1.5, 2.3, 3, 3] < np.array([-1.0, 4.0, 3, 2]))

        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )

        z0 = runtime.read_value_from_storage("alice", "z0")
        z1 = runtime.read_value_from_storage("alice", "z1")
        z2 = runtime.read_value_from_storage("alice", "z2")

        np.testing.assert_equal(z0, z[0])
        np.testing.assert_equal(z1, z[1])
        np.testing.assert_equal(z2, z[2])

        np.testing.assert_equal(
            runtime.read_value_from_storage("alice", "less_result"), z
        )

        np.testing.assert_equal(runtime.read_value_from_storage("alice", "y0"), 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
