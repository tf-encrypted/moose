from absl.testing import parameterized

from moose.computation.utils import deserialize_computation
from moose.computation.utils import serialize_computation
from moose.edsl import base as edsl
from moose.edsl.tracer import trace


class SerdeTest(parameterized.TestCase):
    def test_serde(self):
        alice = edsl.host_placement("alice")
        bob = edsl.host_placement("bob")
        carole = edsl.host_placement("carole")
        replicated = edsl.replicated_placement("rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp():
            x = edsl.constant(1, placement=alice)
            y = edsl.constant(2, placement=bob)
            z = edsl.add(x, y, placement=replicated)
            v = edsl.add(z, z, placement=carole)
            return v

        original = trace(my_comp, compiler_passes=[])
        serialized = serialize_computation(original)
        deserialized = deserialize_computation(serialized)

        assert deserialized == original
