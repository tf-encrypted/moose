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
    def _setup_comp(self, dtype, input_placement, reshape_placement):
        @edsl.computation
        def my_comp():
            with input_placement:
                x = edsl.constant(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=dtype)
                # shape = edsl.constant(np.array([[1., 1., 1., 1.]]), dtype=dtype)

            with reshape_placement:
                # shape = edsl.shape(shape)
                # x_reshape = edsl.reshape(x, shape)
                x_reshape = edsl.reshape(x, [1, 4])

            with input_placement:
                res = edsl.save("x_reshape", x_reshape)

            return res

        return my_comp

    @parameterized.parameters(
        (edsl.float64, bob),
        # (edsl.fixed(14, 23), bob),
        # (edsl.fixed(24, 40), bob),
        # (edsl.fixed(14, 23), rep),
        # (edsl.fixed(24, 40), rep),
    )
    def test_example_execute(self, dtype, shape_placement):
        comp = self._setup_comp(dtype, alice, shape_placement)
        traced_comp = edsl.trace(comp)

        storage = {
            "alice": {},
            "carole": {},
            "bob": {},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        results = runtime.evaluate_computation(
            computation=traced_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        res_array = runtime.read_value_from_storage("alice", "x_reshape")

        assert res_array.shape == (1, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
