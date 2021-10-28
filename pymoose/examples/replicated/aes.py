import argparse
import logging
import unittest

import numpy as np

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


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

        concrete_comp = edsl.trace(my_comp)
        comp_bin = utils.serialize_computation(concrete_comp)
        rust_compiled = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                # "dump",
                "full",
                "prune",
                "networking",
                "typing",
                # "dump",
                # "print",
            ],
        )
        runtime.evaluate_compiled(
            comp_bin=rust_compiled,
            role_assignment={
                "alice": "alice",
                "bob": "bob",
                "carole": "carole",
                "dave": "dave",
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
