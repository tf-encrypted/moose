import argparse
import logging
import unittest

import numpy as np

from moose import edsl
from moose.logger import get_logger
from moose.testing import TestRuntime as Runtime


class FixedpointExample(unittest.TestCase):
    def test_fixedpoint_example(self):

        alice = edsl.host_placement(name="alice")

        @edsl.computation
        def my_comp():

            with alice:
                x = edsl.constant(np.array([10.0, 12.0]), dtype=edsl.fixed(8, 27))
                y = edsl.mul(x, x)
                interm = edsl.save("interm", y)
                z = edsl.cast(y, dtype=edsl.float64)
                res = edsl.save("res", z)

            return interm, res

        concrete_comp = edsl.trace(my_comp, render=True)

        runtime = Runtime()
        runtime.evaluate_computation(
            concrete_comp, placement_instantiation={alice: "worker0"},
        )
        print(runtime.existing_executors["worker0"].storage.store["interm"])
        print(runtime.existing_executors["worker0"].storage.store["res"])

        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
