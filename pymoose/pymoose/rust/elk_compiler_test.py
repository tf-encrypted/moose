import argparse
import logging

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from pymoose import edsl
from pymoose import elk_compiler
from pymoose import LocalRuntime
from pymoose.computation import utils
from pymoose.logger import get_logger

_alice = edsl.host_placement(name="alice")
_bob = edsl.host_placement(name="bob")
_carole = edsl.host_placement("carole")
_rep = edsl.replicated_placement("replicated", [_alice, _bob, _carole])
_fpd = edsl.fixed(24, 40)
_DEFAULT_PASSES = [
    "typing",
    "full",
    "prune",
    "networking",
    "toposort",
]

@edsl.computation
def _reference_computation():
    with _alice:
        a = edsl.constant(np.array([[1.0], [2.0]], dtype=np.float64), dtype=_fpd)
    with _bob:
        b = edsl.constant(np.array([[3.0], [4.0]], dtype=np.float64), dtype=_fpd)
    with _rep:
        c = edsl.mul(a, b)
    with _carole:
        c = edsl.cast(c, edsl.float64)
    return c


class CompileComputation(parameterized.TestCase):
    def setUp(self):
        self.empty_storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        self.role_assignment = {
            "alice": "alice",
            "bob": "bob",
            "carole": "carole",
        }

    def test_serde_only(self):
        self._trace_and_compile(passes=[])

    def test_successful_compilation(self):
        self._trace_and_compile(passes=None)

    def test_default_passes_arg(self):
        comp0 = self._trace_and_compile(passes=None)
        comp1 = self._trace_and_compile(passes=_DEFAULT_PASSES)
        res0 = self._build_new_runtime().evaluate_compiled(comp0, self.role_assignment, {})
        res1 = self._build_new_runtime().evaluate_compiled(comp1, self.role_assignment, {})
        np.testing.assert_equal(res0, res1)

    def _build_new_runtime(self):
        return LocalRuntime(self.empty_storage)

    def _trace_and_compile(self, passes):
        traced = edsl.trace(_reference_computation)
        pyserialized = utils.serialize_computation(traced)
        rustref = elk_compiler.compile_computation(pyserialized, passes=passes)
        return rustref

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="elk compiler tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
