from absl.testing import parameterized

from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import host_placement
from moose.edsl.base import replicated_placement
from moose.edsl.base import share
from moose.edsl.tracer import trace


class ReplicatedTest(parameterized.TestCase):
    def test_replicated(self):
        alice = host_placement(name="alice")
        bob = host_placement(name="bob")
        carole = host_placement(name="carole")
        replicated = replicated_placement("replicated", players=[alice, bob, carole])
        # dave = host_placement(name="dave")

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

        concrete_comp = trace(my_comp)
        del concrete_comp

        # assert False
