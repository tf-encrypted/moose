import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime

alice = pm.host_placement(name="alice")
bob = pm.host_placement(name="bob")
carole = pm.host_placement(name="carole")
mir = pm.mirrored_placement(name="mir", players=[alice, bob, carole])
rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])


class ShapeExample(parameterized.TestCase):
    def _setup_comp(self, dtype, input_placement, shape_placement):
        @pm.computation
        def my_comp():
            with input_placement:
                x = pm.constant(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=dtype)

            with shape_placement:
                x_shape = pm.shape(x)

            with input_placement:
                res = pm.ones(x_shape, pm.float64)

            return res

        return my_comp

    @parameterized.parameters(
        (pm.float64, bob),
        (pm.fixed(14, 23), bob),
        (pm.fixed(24, 40), bob),
        (pm.fixed(14, 23), rep),
        (pm.fixed(24, 40), rep),
    )
    def test_example_execute(self, dtype, shape_placement):
        comp = self._setup_comp(dtype, alice, shape_placement)
        traced_comp = pm.trace(comp)

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
        res_array = list(results.values())[0]
        assert res_array.shape == (2, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
