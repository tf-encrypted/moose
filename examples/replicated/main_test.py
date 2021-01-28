import argparse
import logging
import unittest

import numpy as np

from moose import edsl
from moose.edsl import add
from moose.edsl import computation
from moose.edsl import constant
from moose.edsl import dot
from moose.edsl import host_placement
from moose.edsl import mul
from moose.edsl import replicated_placement
from moose.edsl import save
from moose.edsl import trace
from moose.logger import get_logger
from moose.testing import TestRuntime as Runtime


class ReplicatedExample(unittest.TestCase):
    def test_replicated_example(self):

        alice = host_placement(name="alice")
        bob = host_placement(name="bob")
        carole = host_placement(name="carole")
        dave = host_placement(name="dave")
        eric = host_placement(name="eric")
        rep = replicated_placement(name="rep", players=[alice, bob, carole])

        @computation
        def my_comp():

            with alice:
                x = constant(np.array([1, 2], dtype=np.float64))

            with bob:
                y = constant(np.array([1, 1], dtype=np.float64))

            with rep:
                # z1 = mul(x, y)
                # z2 = dot(x, y)
                c = edsl.print(y)

            # with dave:
            #     v = add(z1, z1)
            #     res_dave = save("res", v)

            # with eric:
            #     w = add(z1, z1)
            #     res_eric = save("res", w)

            # return (res_dave, res_eric)
            return c

        print("my comp: ", my_comp)
        concrete_comp = trace(my_comp, render=True)

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
