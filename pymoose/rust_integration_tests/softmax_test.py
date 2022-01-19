import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation.standard import IntType
from pymoose.computation.standard import StringType
from pymoose.edsl.base import index_axis
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class SoftmaxExample(parameterized.TestCase):
    def _setup_comp(self, axis, axis_idx_max):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp(x_uri: edsl.Argument(placement=bob, vtype=StringType()),):
            with bob:
                x = edsl.load(x_uri, dtype=edsl.float64)
                x_fixed = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with rep:
                x_soft = edsl.softmax(x_fixed, axis=axis, upmost_index=axis_idx_max)

            with bob:
                x_soft_host = edsl.cast(x_soft, dtype=edsl.float64)
                res = edsl.save("softmax", x_soft_host)

            return res

        return my_comp

    @parameterized.parameters(
        ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], 0, 2),
        ([[[1, 10, 1], [4, 10, 3]], [[1, 5, 1], [3, 4, 3]]], 0, 2),
    )
    def test_example_execute(self, x, axis, axis_idx_max):
        comp = self._setup_comp(axis, axis_idx_max)
        traced_less_comp = edsl.trace(comp)

        x_arg = np.array(x, dtype=np.float64)
        print("XXXXXXXXXXXXXXXX axis=", axis)

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

        x0 = runtime.read_value_from_storage("bob", "softmax")

        # e_x = np.exp(x_arg - x_arg.max(axis=axis))
        # actual_softmax = e_x / e_x.sum(axis=axis)

        print("XXXXXXXXXX\n", x_arg - x_arg.max(axis=axis))
        actual_softmax = np.exp(x_arg - x_arg.max(axis=axis))

        np.testing.assert_almost_equal(x0, actual_softmax, decimal=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
