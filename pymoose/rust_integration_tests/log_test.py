import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger


class ReplicatedExample(parameterized.TestCase):
    def _setup_log_comp(self, x_array, log_op):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_exp_comp():
            with bob:
                x = pm.constant(x_array)
                x_enc = pm.cast(x, dtype=pm.fixed(8, 27))

            with rep:
                y = log_op(x_enc)

            with alice:
                res = pm.save("y_uri", pm.cast(y, pm.float64))

            return res

        return my_exp_comp

    @parameterized.parameters(
        ([1, 3, 2, 3], pm.log, np.log),
        ([1.32, 10.42, 2.321, 3.5913], pm.log, np.log),
        ([4.132, 1.932, 2, 4.5321], pm.log, np.log),
        ([1, 2, 4, 8, 4.5, 10.5], pm.log2, np.log2),
        (
            [[1.0, 2.0], [4.0, 23.3124], [42.954, 4.5], [10.5, 13.4219]],
            pm.log2,
            np.log2,
        ),
    )
    def test_log_example_execute(self, x, log_op, np_log):
        x_arg = np.array(x, dtype=np.float64)
        exp_comp = self._setup_log_comp(x_arg, log_op)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = pm.LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=exp_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        np.testing.assert_almost_equal(actual_result, np_log(x_arg), decimal=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
