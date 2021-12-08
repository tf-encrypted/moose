import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class BooleanLogicExample(parameterized.TestCase):
    def _setup_less_comp(self, include_mux):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_bool_comp():
            with bob:
                x = edsl.constant(np.array([1.5, 2.3, 3, 3], dtype=np.float64))
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

                y = edsl.constant(np.array([-1.0, 4.0, 3, 2], dtype=np.float64))
                y = edsl.cast(y, dtype=edsl.fixed(8, 27))

            with rep:
                z_rep = edsl.less(x, y)
                if include_mux:
                    z_rep = edsl.mux(z_rep, x, y)

            with alice:
                if include_mux:
                    z_host = edsl.cast(z_rep, dtype=edsl.float64)
                else:
                    z_host = edsl.logical_or(z_rep, z_rep)

            return z_host

        return my_bool_comp

    @parameterized.parameters(
        (False, np.array([1.5, 2.3, 3, 3]) < np.array([-1.0, 4.0, 3, 2])),
        (True, np.array([-1.0, 2.3, 3, 2])))
    def test_boolean_execute(self, include_mux, expected):
        boolean_comp = self._setup_less_comp(include_mux)
        traced_bool_comp = edsl.trace(boolean_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        comp_result = runtime.evaluate_computation(
            computation=traced_bool_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        np.testing.assert_almost_equal(list(comp_result.values())[0], expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
