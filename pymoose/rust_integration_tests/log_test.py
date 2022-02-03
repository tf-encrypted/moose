import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_log_comp(self, x_array):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_exp_comp():
            with bob:
                x = edsl.constant(x_array)
                x_enc = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with rep:
                y = edsl.log(x_enc)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_exp_comp

    @parameterized.parameters(
        ([1, 3, 2, 3],), ([1.32, 10.42, 2.321, 3.5913],), ([4.132, 1.932, 2, 4.5321],),
    )
    def test_exp_example_execute(self, x):
        x_arg = np.array(x, dtype=np.float64)
        exp_comp = self._setup_log_comp(x_arg)
        traced_exp_comp = edsl.trace(exp_comp)
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
        np.testing.assert_almost_equal(actual_result, np.log(x_arg), decimal=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
