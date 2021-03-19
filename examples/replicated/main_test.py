import argparse
import logging
import unittest

import numpy as np

from moose import edsl
from moose.logger import get_logger
from moose.testing import TestRuntime as Runtime


class ReplicatedExample(unittest.TestCase):
    def test_replicated_example(self):

        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        dave = edsl.host_placement(name="dave")
        eric = edsl.host_placement(name="eric")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp():

            with alice:
                x = edsl.constant(np.array([1, 2], dtype=np.float64))

            with bob:
                y = edsl.constant(np.array([1, 1], dtype=np.float64))

            with rep:
                z1 = edsl.mul(x, y)
                z2 = edsl.dot(x, y)
                c = edsl.abs(z2)

            with dave:
                v = edsl.add(z1, z1)
                res_dave = edsl.save("res", v)
                abs_dave = edsl.save("abs", c)

            with eric:
                w = edsl.add(z2, z2)
                res_eric = edsl.save("res", w)

            return (res_dave, abs_dave, res_eric)

        concrete_comp = edsl.trace(my_comp)

        runtime = Runtime()
        runtime.evaluate_computation(
            concrete_comp,
            placement_instantiation={
                alice: "worker0",
                bob: "worker1",
                carole: "worker2",
                dave: "worker3",
                eric: "worker4",
            },
        )

        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
