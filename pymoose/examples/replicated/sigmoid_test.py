import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_sigmoid_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_sigmoid_comp():
            with bob:
                x = edsl.constant(np.array([2], dtype=np.float64))
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with rep:
                y = edsl.sigmoid(x)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_sigmoid_comp

    def test_sigmoid_example_serde(self):
        sigmoid_comp = self._setup_sigmoid_comp()
        traced_sigmoid_comp = edsl.trace(sigmoid_comp)
        comp_bin = utils.serialize_computation(traced_sigmoid_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    def test_sigmoid_example_compile(self):
        sigmoid_comp = self._setup_sigmoid_comp()
        traced_sigmoid_comp = edsl.trace(sigmoid_comp)
        comp_bin = utils.serialize_computation(traced_sigmoid_comp)
        _ = elk_compiler.compile_computation(comp_bin)

    def test_sigmoid_example_execute(self):
        sigmoid_comp = self._setup_sigmoid_comp()
        traced_sigmoid_comp = edsl.trace(sigmoid_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_sigmoid_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")

        def exp_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        np.testing.assert_almost_equal(actual_result, exp_sigmoid(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
