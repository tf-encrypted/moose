from absl.testing import parameterized

from moose.computation.utils import deserialize_computation
from moose.computation.utils import serialize_computation
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import host_placement
from moose.edsl.base import replicated_placement
from moose.edsl.tracer import trace


class SerdeTest(parameterized.TestCase):
    def test_serde(self):
        alice = host_placement("alice")
        bob = host_placement("bob")
        carole = host_placement("carole")
        replicated = replicated_placement("rep", players=[alice, bob, carole])

        @computation
        def my_comp():
            x = constant(1, placement=alice)
            y = constant(2, placement=bob)
            z = add(x, y, placement=replicated)
            v = add(z, z, placement=carole)
            return v

        original = trace(my_comp, compiler_passes=[])
        serialized = serialize_computation(original)
        deserialized = deserialize_computation(serialized)

        assert deserialized == original
