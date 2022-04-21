import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import types as ty
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class SliceExample(parameterized.TestCase):
    def _setup_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp(
            x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
        ):
            with bob:
                x = edsl.load(x_uri, dtype=edsl.float64)
                xs = edsl.better_slice(
                    x, (slice(None), slice(1, None, None), slice(1, None, None))
                )
                res = (edsl.save("sliced", xs),)

            return res

        return my_comp

    @parameterized.parameters(([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],))
    def test_example_execute(self, x):
        comp = self._setup_comp()
        traced_sliced_comp = edsl.trace(comp)

        x_arg = np.array(x, dtype=np.float64)

        storage = {
            "alice": {},
            "carole": {},
            "bob": {"x_arg": x_arg},
        }

        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_sliced_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x_uri": "x_arg"},
        )

        x_sliced = runtime.read_value_from_storage("bob", "sliced")
        np.testing.assert_equal(x_sliced, x_arg[::, 1::, 1::])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
