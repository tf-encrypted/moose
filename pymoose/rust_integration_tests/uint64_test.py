import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_int64_comp(self, x_array):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_int_comp():
            with bob:
                x = edsl.constant(x_array)
                x_ring = edsl.cast(x, dtype=edsl.ring64)

            with rep:
                y = edsl.add(x_ring, x_ring)

            with alice:
                y_uint = edsl.cast(y, dtype=edsl.uint64)
                res = edsl.save("y_uri", y_uint)

            return res

        return my_int_comp

    @parameterized.parameters(([1, 3, 2, 3],),)
    def test_int_example_execute(self, x):
        x_arg = np.array(x, dtype=np.uint64)
        int_comp = self._setup_int64_comp(x_arg)
        traced_exp_comp = edsl.trace(int_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_exp_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        np.testing.assert_equal(actual_result, x_arg * 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
