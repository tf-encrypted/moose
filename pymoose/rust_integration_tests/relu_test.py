import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_relu_comp(self, x_array):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_relu_comp():
            with bob:
                x = pm.constant(x_array)
                x_enc = pm.cast(x, dtype=pm.fixed(8, 27))

            with rep:
                y = pm.relu(x_enc)

            with alice:
                res = pm.save("y_uri", pm.cast(y, pm.float64))

            return res

        return my_relu_comp

    @parameterized.parameters(
        ([-1, -2, 0, 2, 3],),
        ([-1.25, -2.84, 0.001, 2.39, 3.12],),
        ([10.1, 8.3, 1.1],),
        ([-0.01, -0.03],),
        ([0.02, -0.04, 1.08, -1.38, 3.65, -1.56],),
        ([-0.71, 2.3, -0.74, 0.02, -0.04, 1.08, -0.64, 0.76, 0.97],),
    )
    def test_relu_example_execute(self, x):
        x_arg = np.array(x, dtype=np.float64)
        relu_comp = self._setup_relu_comp(x_arg)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=relu_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")

        def relu(x):
            return x * (x > 0)

        np.testing.assert_almost_equal(actual_result, relu(x_arg), decimal=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relu example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
