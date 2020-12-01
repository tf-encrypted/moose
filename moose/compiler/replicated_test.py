from absl.testing import parameterized

from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import mul
from moose.edsl.tracer import trace


class ReplicatedTest(parameterized.TestCase):
    def test_replicated(self):
        alice = HostPlacement(name="alice")
        bob = HostPlacement(name="bob")
        carole = HostPlacement(name="carole")
        replicated = ReplicatedPlacement("replicated", alice, bob, carole)
        dave = HostPlacement(name="dave")

        @computation
        def my_comp():
            x = constant(1, placement=alice)
            y = constant(2, placement=bob)
            z = add(x, y, placement=replicated)
            v = constant(3, placement=dave)
            w = add(z, v, placement=dave)
            return w

        concrete_comp = trace(my_comp, render=True)
        del concrete_comp

        assert False
