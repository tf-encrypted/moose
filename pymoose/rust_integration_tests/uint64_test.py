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

        @edsl.computation
        def my_int_comp():
            with bob:
                x = edsl.constant(x_array)

            with alice:
                x_alice = edsl.identity(x)
                res = edsl.save("x_uri", x_alice)

            return res

        return my_int_comp

    @parameterized.parameters(
        ([1, 3, 2, 3],),
    )
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
        actual_result = runtime.read_value_from_storage("alice", "x_uri")
        np.testing.assert_equal(actual_result, x_arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uint64 example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
