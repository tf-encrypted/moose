from absl.testing import parameterized

from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import mul
from moose.edsl.base import share
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
            # TODO `share` and `reconstruct` added by pass
            x_bar = share(x, placement=replicated)
            y_bar = share(y, placement=replicated)
            z_bar = add(x_bar, y_bar, placement=replicated)
            # z = reconstruct(z_bar, placement=replicated)
            # v = constant(3, placement=dave)
            # w = add(z, v, placement=dave)
            # return w
            return z_bar

        concrete_comp = trace(my_comp, render=True)
        del concrete_comp

        assert False
