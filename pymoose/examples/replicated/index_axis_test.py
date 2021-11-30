import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.computation.standard import TensorType
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_index_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_model_comp(x: edsl.Argument(bob, vtype=TensorType(edsl.float64)),):
            with bob:
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))

            with rep:
                y = edsl.index_axis(x, axis=0, index=1)

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_model_comp

    def test_logistic_regression_example_serde(self):
        model_comp = self._setup_index_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        deser_model_comp = utils.deserialize_computation(comp_bin)
        assert traced_model_comp == deser_model_comp

    def test_logistic_regression_example_rust_serde(self):
        model_comp = self._setup_index_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    def test_logistic_regression_example_compile(self):
        model_comp = self._setup_index_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        _ = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                "toposort",
                # "print",
            ],
        )

    def test_index_axis_example_execute(self):
        input_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        expected_result = np.array([3.0, 4.0])

        index_comp = self._setup_index_comp()
        traced_index_comp = edsl.trace(index_comp)
        comp_bin = utils.serialize_computation(traced_index_comp)
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
        _ = runtime.evaluate_compiled(
            comp_bin=compiled_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={"x": input_x},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")

        np.testing.assert_almost_equal(actual_result, expected_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
