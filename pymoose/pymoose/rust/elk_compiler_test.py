import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import LocalRuntime
from pymoose import edsl
from pymoose import elk_compiler
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
        self._trace_and_compile()

    def _build_new_runtime(self):
        return LocalRuntime(self.empty_storage)

    def _trace_and_compile(self, passes=None):
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
