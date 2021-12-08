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
    def _setup_exp_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_exp_comp():
            with bob:
                x = edsl.constant(np.array([2], dtype=np.float64))
                x_enc = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with alice:
                x = edsl.identity(x)
                x_enc = edsl.identity(x_enc)

            with rep:
                xe = edsl.identity(x_enc)
                y = edsl.exp(x_enc)

            with alice:
                xe = edsl.cast(xe, edsl.float64)
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res, xe

        return my_exp_comp

    def test_exp_example_serde(self):
        exp_comp = self._setup_exp_comp()
        traced_exp_comp = edsl.trace(exp_comp)
        comp_bin = utils.serialize_computation(traced_exp_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    def test_exp_example_compile(self):
        exp_comp = self._setup_exp_comp()
        traced_exp_comp = edsl.trace(exp_comp)
        comp_bin = utils.serialize_computation(traced_exp_comp)
        _ = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                "toposort",
                # "print",
            ],
        )

    def test_exp_example_execute(self):
        exp_comp = self._setup_exp_comp()
        traced_exp_comp = edsl.trace(exp_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_exp_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")
        np.testing.assert_almost_equal(actual_result, np.exp([2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
