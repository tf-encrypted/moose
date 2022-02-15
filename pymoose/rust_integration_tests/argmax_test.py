import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation.standard import StringType
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ArgmaxExample(parameterized.TestCase):
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
                x_arg = edsl.argmax(x_fixed, axis=axis, upmost_index=axis_idx_max)

            with bob:
                x_arg_host = edsl.identity(x_arg)
                res = edsl.save("argmax", x_arg_host)

            return res

        return my_comp

    @parameterized.parameters(
        ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], 0, 2),
        ([[[1, 10, 1], [4, 10, 3]], [[1, 5, 1], [3, 4, 3]]], 0, 2),
        ([-142.412, 0, 0, 0, 0], 0, 5),
        (
            [
                [0.90102809, 0.65720883, -0.02816407, 0.0535739],
                [0.61216721, 0.20281131, 1.7734221, -0.69106256],
                [-0.08150293, -1.50330937, -0.99238243, -2.65759917],
            ],
            1,
            4,
        ),
        (
            [
                [0.74997993, 0.6130942, 0.76297858, 0.30081999],
                [-0.64102755, 0.60816909, 0.17821022, 0.6271898],
                [0.72588823, 0.35806336, -0.59945702, 0.6398326],
                [-0.61964023, -0.9119955, 1.50079676, -1.46759315],
            ],
            1,
            4,
        ),
    )
    def test_example_execute(self, x, axis, axis_idx_max):
        comp = self._setup_comp(axis, axis_idx_max)
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

        softmax_runtime = runtime.read_value_from_storage("bob", "argmax")

        np.testing.assert_equal(softmax_runtime, np.argmax(x_arg, axis=axis))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
