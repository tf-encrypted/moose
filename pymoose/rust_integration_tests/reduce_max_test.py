import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation import types as ty
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReducemaxLogicExample(parameterized.TestCase):
    def _setup_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp(x_uri: edsl.Argument(placement=bob, vtype=ty.StringType())):
            with bob:
                x = edsl.load(x_uri, dtype=edsl.float64)
                x_fixed = edsl.cast(x, dtype=edsl.fixed(8, 27))
                x0 = edsl.index_axis(x_fixed, axis=2, index=0)
                x1 = edsl.index_axis(x_fixed, axis=2, index=1)
                x2 = edsl.index_axis(x_fixed, axis=2, index=2)

            with rep:
                x_max = edsl.maximum([x0, x1, x2])

            with bob:
                x_max_host = edsl.cast(x_max, dtype=edsl.float64)
                res = edsl.save("reduce_max", x_max_host)

            return res

        return my_comp

    @parameterized.parameters(
        ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],),
        ([[[1, 10, 1], [4, 100, 32]], [[123, 521, 132], [312, 421, 321]]],),
    )
    def test_example_execute(self, x):
        comp = self._setup_comp()
        traced_less_comp = edsl.trace(comp)

        x_arg = np.array(x, dtype=np.float64)

        storage = {
            "alice": {},
            "carole": {},
            "bob": {"x_arg": x_arg},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg"},
        )

        x0 = runtime.read_value_from_storage("bob", "reduce_max")

        np.testing.assert_almost_equal(x0, x_arg.max(axis=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
