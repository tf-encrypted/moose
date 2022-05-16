import argparse
import logging
import unittest

import numpy as np

import pymoose as pm
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime

FIXED = pm.fixed(14, 23)


class ReplicatedExample(unittest.TestCase):
    def test_replicated_example(self):

        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        dave = pm.host_placement(name="dave")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        @pm.computation
        def my_comp():

            with alice:
                x = pm.constant(np.array([1, 2], dtype=np.float64))
                x = pm.cast(x, dtype=FIXED)

            with bob:
                y = pm.constant(np.array([2, 2], dtype=np.float64))
                y = pm.cast(y, dtype=FIXED)

            with rep:
                z1 = pm.div(x, y)

            with dave:
                z1 = pm.cast(z1, dtype=pm.float64)
                res_dave = pm.save("res", z1)

            return res_dave

        executors_storage = {
            "alice": {},
            "bob": {},
            "carole": {},
            "dave": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=executors_storage)

        logical_comp = pm.trace(my_comp)
        runtime.evaluate_computation(
            computation=logical_comp,
            role_assignment={
                "alice": "alice",
                "bob": "bob",
                "carole": "carole",
                "dave": "dave",
            },
            arguments={},
        )

        print("Done", runtime.read_value_from_storage("dave", "res"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
