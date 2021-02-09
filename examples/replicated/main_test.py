import argparse
import logging
import unittest

import numpy as np

import moose as moo
from moose.logger import get_logger
from moose.testing import TestRuntime as Runtime


class ReplicatedExample(unittest.TestCase):
    def test_replicated_example(self):

        alice = moo.host_placement(name="alice")
        bob = moo.host_placement(name="bob")
        carole = moo.host_placement(name="carole")
        dave = moo.host_placement(name="dave")
        eric = moo.host_placement(name="eric")
        rep = moo.replicated_placement(name="rep", players=[alice, bob, carole])

        @moo.computation
        def my_comp():

            with alice:
                x = moo.constant(np.array([1, 2], dtype=np.float64))

            with bob:
                y = moo.constant(np.array([1, 1], dtype=np.float64))

            with rep:
                z1 = moo.mul(x, y)
                z2 = moo.dot(x, y)
                c = moo.abs(z2)

            with dave:
                v = moo.add(z1, z1)
                res_dave = moo.save("res", v)
                abs_dave = moo.save("abs", c)

            with eric:
                w = moo.add(z2, z2)
                res_eric = moo.save("res", w)

            return (res_dave, abs_dave, res_eric)

        concrete_comp = moo.trace(my_comp)

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
