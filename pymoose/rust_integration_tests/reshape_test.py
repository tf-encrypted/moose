import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime

alice = edsl.host_placement(name="alice")
bob = edsl.host_placement(name="bob")
carole = edsl.host_placement(name="carole")
mir = edsl.mirrored_placement(name="mir", players=[alice, bob, carole])
rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])


class ReshapeExample(parameterized.TestCase):
    def _setup_rep_comp(self):
        @edsl.computation
        def my_comp():
            with bob:
                x = edsl.constant(
                    np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=edsl.fixed(8, 27)
                )

            with rep:
                x_reshape = edsl.reshape(x, [1, 4])

            with bob:
                x_reshape = edsl.cast(x_reshape, dtype=edsl.float64)
                res = edsl.save("x_reshape", x_reshape)

            return res

        return my_comp

    def _setup_host_comp(self):
        @edsl.computation
        def my_comp():
            with bob:
                x = edsl.constant(
                    np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=edsl.float64
                )
                x_reshape = edsl.reshape(x, [1, 4])
                res = edsl.save("x_reshape", x_reshape)

            return res

        return my_comp

    @parameterized.parameters(
        (bob),
        (rep),
    )
    def test_example_execute(self, reshape_placement):
        if reshape_placement == bob:
            comp = self._setup_host_comp()
        elif reshape_placement == rep:
            comp = self._setup_rep_comp()

        traced_comp = edsl.trace(comp)

        storage = {
            "alice": {},
            "carole": {},
            "bob": {},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        runtime.evaluate_computation(
            computation=traced_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        res_array = runtime.read_value_from_storage("bob", "x_reshape")

        assert res_array.shape == (1, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshape example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
