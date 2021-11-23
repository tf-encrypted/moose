import argparse
import logging

import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_model_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_model_comp():
            with bob:
                x = edsl.constant(np.array([2], dtype=np.float64))
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))
                w = edsl.constant(np.array([0.5], dtype=np.float64))
                w = edsl.cast(w, dtype=edsl.fixed(8, 27))

            with rep:
                y = edsl.sigmoid(edsl.dot(x, w))

            with alice:
                res = edsl.save("y_uri", edsl.cast(y, edsl.float64))

            return res

        return my_model_comp

    def test_logistic_regression_example_serde(self):
        model_comp = self._setup_model_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        deser_model_comp = utils.deserialize_computation(comp_bin)
        assert traced_model_comp == deser_model_comp

    def test_logistic_regression_example_rust_serde(self):
        model_comp = self._setup_model_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    def test_logistic_regression_example_compile(self):
        model_comp = self._setup_model_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        _ = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                # "print",
            ],
        )

    @pytest.mark.slow
    def test_logistic_regression_example_execute(self):
        model_comp = self._setup_model_comp()
        traced_model_comp = edsl.trace(model_comp)
        comp_bin = utils.serialize_computation(traced_model_comp)
        compiled_comp = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                "prune",
                "networking",
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
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "y_uri")

        def logistic_regression(input, weights):
            y = np.dot(input, weights)
            sigmoid = 1 / (1 + np.exp(-y))
            return sigmoid

        np.testing.assert_almost_equal(actual_result, logistic_regression(np.array([2]), np.array([0.5])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
