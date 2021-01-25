import argparse
import logging
import unittest

import numpy as np

from moose.edsl import computation
from moose.edsl import constant
from moose.edsl import host_placement
from moose.edsl import mul
from moose.edsl import replicated_placement
from moose.edsl import save
from moose.edsl import trace
from moose.logger import get_logger
from moose.testing import TestRuntime as Runtime


class ReplicatedConsecutiveMulExample(unittest.TestCase):
    def test_replicated_consecutive_mul_example(self):

        alice = host_placement(name="alice")
        bob = host_placement(name="bob")
        carole = host_placement(name="carole")
        dave = host_placement(name="dave")
        rep = replicated_placement(name="rep", players=[alice, bob, carole])

        @computation
        def my_comp():

            with alice:
                x = constant(np.array([2], dtype=np.float64))

            with bob:
                y = constant(np.array([3], dtype=np.float64))

            with rep:
                mul_1 = mul(x, y)
                mul_2 = mul(mul_1, mul_1)

            with dave:
                mul_1_dave = save("dave_mul_1", mul_1)
                mul_2_dave = save("dave_mul_2", mul_2)

            return (mul_1_dave, mul_2_dave)

        concrete_comp = trace(my_comp, render=False)

        runtime = Runtime()
        runtime.evaluate_computation(
            concrete_comp,
            placement_instantiation={
                alice: "worker0",
                bob: "worker1",
                carole: "worker2",
                dave: "worker3",
            },
        )

        print("Done")

        output = runtime.existing_executors["worker3"].storage.store
        np.testing.assert_array_equal(output["dave_mul_1"], np.array([6.0]))
        np.testing.assert_array_equal(output["dave_mul_2"], np.array([36.0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
