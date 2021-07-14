import argparse
import logging
import unittest

import numpy as np

from moose import edsl
from moose.logger import get_logger
from moose.testing import LocalMooseRuntime


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
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with bob:
                y = edsl.constant(np.array([1, 1], dtype=np.float64))
                y = edsl.cast(y, dtype=edsl.fixed(8, 27))

            with rep:
                z1 = edsl.mul(x, y)
                z2 = edsl.dot(x, y)
                c = edsl.abs(z2)

            with dave:
                z1 = edsl.cast(z1, dtype=edsl.float64)
                c = edsl.cast(c, dtype=edsl.float64)
                v = edsl.add(z1, z1)
                res_dave = edsl.save("res", v)
                abs_dave = edsl.save("abs", c)

            with eric:
                z2 = edsl.cast(z2, dtype=edsl.float64)
                w = edsl.add(z2, z2)
                res_eric = edsl.save("res", w)

            return (res_dave, abs_dave, res_eric)

        executors_storage = {
            "alice": {},
            "bob": {},
            "carole": {},
            "dave": {},
            "eric": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=executors_storage)
        runtime.evaluate_computation(
            computation=my_comp,
            role_assignment={
                "alice": "alice",
                "bob": "bob",
                "carole": "carole",
                "dave": "dave",
                "eric": "eric",
            },
            arguments={},
        )

        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
