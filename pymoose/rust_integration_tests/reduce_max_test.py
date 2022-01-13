import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReducemaxLogicExample(parameterized.TestCase):
    def _setup_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp():
            with bob:
                x = edsl.constant(
                    np.array(
                        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                        dtype=np.float64,
                    )
                )
                xf = edsl.cast(x, dtype=edsl.fixed(8, 27))
                res = edsl.save("x0", edsl.index_axis(x, axis=2, index=0))
                x0 = edsl.index_axis(xf, axis=2, index=0)
                x1 = edsl.index_axis(xf, axis=2, index=1)
                x2 = edsl.index_axis(xf, axis=2, index=2)

            with rep:
                x_max = edsl.maximum([x0, x1, x2, x0])

            with bob:
                x_max_host = edsl.cast(x_max, dtype=edsl.float64)

            return res, x_max_host

        return my_comp

    def test_example_execute(self):
        comp = self._setup_comp()
        traced_less_comp = edsl.trace(comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        result_dict = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )

        print("outputs: ", result_dict)

        x0 = runtime.read_value_from_storage("bob", "x0")
        np.testing.assert_almost_equal(x0, [[1, 4], [7, 10]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
