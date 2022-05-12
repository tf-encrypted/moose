import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.computation import utils
from pymoose.logger import get_logger

_alice = pm.host_placement(name="alice")
_bob = pm.host_placement(name="bob")
_carole = pm.host_placement("carole")
_rep = pm.replicated_placement("replicated", [_alice, _bob, _carole])
_fpd = pm.fixed(24, 40)


@pm.computation
def _reference_computation():
    with _alice:
        a = pm.constant(np.array([[1.0], [2.0]], dtype=np.float64), dtype=_fpd)
    with _bob:
        b = pm.constant(np.array([[3.0], [4.0]], dtype=np.float64), dtype=_fpd)
    with _rep:
        c = pm.mul(a, b)
    with _carole:
        c = pm.cast(c, pm.float64)
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
        return pm.rust.LocalRuntime(self.empty_storage)

    def _trace_and_compile(self, passes=None):
        traced = pm.trace(_reference_computation)
        pyserialized = utils.serialize_computation(traced)
        rustref = pm.elk_compiler.compile_computation(pyserialized, passes=passes)
        return rustref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="elk compiler tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
