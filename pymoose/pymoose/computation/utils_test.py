import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.computation.utils import deserialize_computation
from pymoose.computation.utils import serialize_computation
from pymoose.edsl import tracer


class SerdeTest(parameterized.TestCase):
    def _build_comp_fixture(self):
        alice = pm.host_placement("alice")
        bob = pm.host_placement("bob")
        carole = pm.host_placement("carole")
        replicated = pm.replicated_placement("rep", players=[alice, bob, carole])

        @pm.computation
        def my_comp():
            x = pm.constant(1, dtype=pm.float32, placement=alice)
            y = pm.constant(2, dtype=pm.float32, placement=bob)
            z = pm.add(x, y, placement=replicated)
            v = pm.add(z, z, placement=carole)
            return v

        return my_comp

    def test_serde(self):
        my_comp = self._build_comp_fixture()
        original = tracer.trace(my_comp)
        serialized = serialize_computation(original)
        deserialized = deserialize_computation(serialized)

        assert deserialized == original

    def test_rust_serde(self):
        my_comp = self._build_comp_fixture()
        traced_comp = tracer.trace(my_comp)
        serialized = serialize_computation(traced_comp)
        # just need to convert PyComputation to pm.MooseComputation,
        # so compile w/ empty passes arg
        moose_comp: pm.MooseComputation = pm.elk_compiler.compile_computation(
            serialized, []
        )
        original_bytes = moose_comp.to_bytes()
        rebuilt_moose_comp = pm.MooseComputation.from_bytes(original_bytes)
        result_bytes = rebuilt_moose_comp.to_bytes()

        assert hash(original_bytes) == hash(result_bytes)

    def test_rust_to_from_disk(self):
        my_comp = self._build_comp_fixture()
        traced_comp = tracer.trace(my_comp)
        serialized = serialize_computation(traced_comp)
        # just need to convert PyComputation to pm.MooseComputation,
        # so compile w/ empty passes arg
        moose_comp: pm.MooseComputation = pm.elk_compiler.compile_computation(
            serialized, []
        )
        original_bytes = moose_comp.to_bytes()
        assert len(original_bytes) > 0

        tempdir = tempfile.gettempdir()
        filepath = os.path.join(tempdir, "temp_comp.moose")
        moose_comp.to_disk(filepath)
        result = pm.MooseComputation.from_disk(filepath)
        result_bytes = result.to_bytes()

        assert hash(original_bytes) == hash(result_bytes)


if __name__ == "__main__":
    absltest.main()
