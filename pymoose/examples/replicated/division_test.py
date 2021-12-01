import argparse
import logging
import unittest

import numpy as np

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime

FIXED = edsl.fixed(14, 23)


class ReplicatedExample(unittest.TestCase):
    def test_replicated_example(self):

        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        dave = edsl.host_placement(name="dave")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp():

            with alice:
                x = edsl.constant(np.array([1, 2], dtype=np.float64))
                x = edsl.cast(x, dtype=FIXED)

            with bob:
                y = edsl.constant(np.array([2, 2], dtype=np.float64))
                y = edsl.cast(y, dtype=FIXED)

            with rep:
                z1 = edsl.div(x, y)

            with dave:
                z1 = edsl.cast(z1, dtype=edsl.float64)
                res_dave = edsl.save("res", z1)

            return res_dave

        executors_storage = {
            "alice": {},
            "bob": {},
            "carole": {},
            "dave": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=executors_storage)

        logical_comp = edsl.trace(my_comp)
        runtime.evaluate_computation(
            computation=logical_comp,
            role_assignment={
                "alice": "alice",
                "bob": "bob",
                "carole": "carole",
                "dave": "dave",
            },
            arguments={},
            compiler_passes=[
                "typing",
                "full",
                "prune",
                "networking",
                "typing",
                "toposort",
            ],
        )

        print("Done", runtime.read_value_from_storage("dave", "res"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
