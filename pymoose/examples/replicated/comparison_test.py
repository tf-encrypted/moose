import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.edsl.tracer import trace
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_less_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_less_comp():
            with bob:
                x = edsl.constant(np.array([1.5, 2.3, 3, 3], dtype=np.float64))
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

                y = edsl.constant(np.array([-1.0, 4.0, 3, 2], dtype=np.float64))
                y = edsl.cast(y, dtype=edsl.fixed(8, 27))

            with rep:
                z_rep = edsl.less(x, y)

            with alice:
                z_host = edsl.bitwise_or(z_rep, z_rep)

            return z_host

        return my_less_comp

    def test_less_example_serde(self):
        less_comp = self._setup_less_comp()
        traced_less_comp = edsl.trace(less_comp)
        print(traced_less_comp)
        comp_bin = utils.serialize_computation(traced_less_comp)
        deser_less_comp = utils.deserialize_computation(comp_bin)
        assert traced_less_comp == deser_less_comp

    def test_less_example_rust_serde(self):
        less_comp = self._setup_less_comp()
        traced_less_comp = edsl.trace(less_comp)
        comp_bin = utils.serialize_computation(traced_less_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    def test_less_example_compile(self):
        less_comp = self._setup_less_comp()
        traced_less_comp = edsl.trace(less_comp)
        comp_bin = utils.serialize_computation(traced_less_comp)
        _ = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                "toposort",
                # "print",
            ],
        )

    def test_less_example_execute(self):
        less_comp = self._setup_less_comp()
        traced_less_comp = edsl.trace(less_comp)
        comp_bin = utils.serialize_computation(traced_less_comp)
        compiled_comp = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                "prune",
                "networking",
                "toposort",
                # "print",
            ],
        )
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        comp_result = runtime.evaluate_compiled(
            comp_bin=compiled_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        real = np.array([1.5, 2.3, 3, 3] < np.array([-1.0, 4.0, 3, 2]))
        np.testing.assert_equal(list(comp_result.values())[0], real)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
