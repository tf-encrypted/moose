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
                xs = edsl.better_slice(x, (slice(None), slice(None)))
                res = (edsl.save("sliced", xs),)

            return res

        return my_comp

    def test_example_execute(self):
        comp = self._setup_comp()
        traced_sliced_comp = edsl.trace(comp)

        print(traced_sliced_comp)
        comp_bin = utils.serialize_computation(traced_sliced_comp)
        _ = elk_compiler.compile_computation(comp_bin)

        # x_arg = np.array(x, dtype=np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
