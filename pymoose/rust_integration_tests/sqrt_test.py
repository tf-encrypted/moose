import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_sqrt_comp(self, x_array):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_sqrt_comp():
            with bob:
                x = pm.constant(x_array)
                x_enc = pm.cast(x, dtype=pm.fixed(8, 27))

            with rep:
                y = pm.sqrt(x_enc)

            with alice:
                res = pm.save("y_uri", pm.cast(y, pm.float64))

            return res

        return my_sqrt_comp

    @parameterized.parameters(
        ([1.0, 2.0, 3.0, 4.0],),
        ([5.0, 6.0, 7.0, 8.0],),
    )
    def test_sqrt_example_execute(self, x):
        x_arg = np.array(x, dtype=np.float64)
        sqrt_comp = self._setup_sqrt_comp(x_arg)
        traced_sqrt_comp = pm.trace(sqrt_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_sqrt_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        np.testing.assert_almost_equal(actual_result, np.sqrt(x_arg), decimal=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sqrt example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
