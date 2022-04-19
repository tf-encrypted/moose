import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_relu_comp(self, x_array):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_relu_comp():
            with bob:
                x = edsl.constant(x_array)
                y = edsl.relu(x)
                # x_enc = edsl.cast(x, dtype=edsl.fixed(8, 27))

            # with rep:
            #     y = edsl.relu(x_enc)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_relu_comp

    @parameterized.parameters(
        ([-1, -2, 0, 2, 3],),
    )
    def test_relu_example_execute(self, x):
        x_arg = np.array(x, dtype=np.float64)
        relu_comp = self._setup_relu_comp(x_arg)
        traced_relu_comp = edsl.trace(relu_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_relu_comp,
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
