from absl.testing import absltest
from absl.testing import parameterized

from pymoose import MooseComputation
from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation.utils import deserialize_computation
from pymoose.computation.utils import serialize_computation
from pymoose.edsl import tracer


class SerdeTest(parameterized.TestCase):
    def _build_comp_fixture(self):
        alice = edsl.host_placement("alice")
        bob = edsl.host_placement("bob")
        carole = edsl.host_placement("carole")
        replicated = edsl.replicated_placement("rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp():
            x = edsl.constant(1, dtype=edsl.float32, placement=alice)
            y = edsl.constant(2, dtype=edsl.float32, placement=bob)
            z = edsl.add(x, y, placement=replicated)
            v = edsl.add(z, z, placement=carole)
            return v

        return my_comp

    def test_serde(self):
        my_comp = self._build_comp_fixture()
        original = tracer.trace_and_compile(my_comp, compiler_passes=[])
        serialized = serialize_computation(original)
        deserialized = deserialize_computation(serialized)

        assert deserialized == original

    def test_rust_serde(self):
        my_comp = self._build_comp_fixture()
        traced_comp = tracer.trace(my_comp)
        serialized = serialize_computation(traced_comp)
        # just need to convert PyComputation to MooseComputation,
        # so compile w/ empty passes arg
        moose_comp: MooseComputation = elk_compiler.compile_computation(serialized, [])
        original_bytes = moose_comp.to_bytes()
        rebuilt_moose_comp = MooseComputation.from_bytes(original_bytes)
        result_bytes = rebuilt_moose_comp.to_bytes()

        assert hash(original_bytes) == hash(result_bytes)


if __name__ == "__main__":
    absltest.main()
