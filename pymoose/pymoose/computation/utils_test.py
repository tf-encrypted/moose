from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation.utils import deserialize_computation
from pymoose.computation.utils import serialize_computation
from pymoose.edsl.tracer import trace_and_compile


class SerdeTest(parameterized.TestCase):
    def test_serde(self):
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

        original = trace_and_compile(my_comp, compiler_passes=[])
        serialized = serialize_computation(original)
        deserialized = deserialize_computation(serialized)

        assert deserialized == original


if __name__ == "__main__":
    absltest.main()
