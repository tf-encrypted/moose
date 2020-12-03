from absl.testing import parameterized

from moose.computation.mpspdz import MpspdzCallOperation
from moose.computation.mpspdz import MpspdzLoadOutputOperation
from moose.computation.mpspdz import MpspdzSaveInputOperation
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import function
from moose.edsl.base import host_placement
from moose.edsl.base import mpspdz_placement
from moose.edsl.base import mul
from moose.edsl.tracer import trace


class MpspdzTest(parameterized.TestCase):
    def test_mpspdz(self):
        alice = host_placement(name="alice")
        bob = host_placement(name="bob")
        carole = host_placement(name="carole")
        mpspdz = mpspdz_placement("mpspdz", players=[alice, bob, carole])

        @function
        def my_function(x, y, z):
            return x * y + z

        @computation
        def my_comp():
            x = constant(1, placement=alice)
            y = constant(2, placement=bob)
            z = constant(3, placement=alice)
            v = my_function(x, y, z, placement=mpspdz)
            w = mul(v, v, placement=carole)
            return w

        concrete_comp = trace(my_comp)

        save_ops = concrete_comp.find_operations_of_type(MpspdzSaveInputOperation)
        assert len(save_ops) == 2
        assert set(op.placement_name for op in save_ops) == {alice.name, bob.name}

        call_ops = concrete_comp.find_operations_of_type(MpspdzCallOperation)
        assert len(call_ops) == 3
        assert set(op.placement_name for op in call_ops) == {
            alice.name,
            bob.name,
            carole.name,
        }

        load_ops = concrete_comp.find_operations_of_type(MpspdzLoadOutputOperation)
        assert len(load_ops) == 1
        assert set(op.placement_name for op in load_ops) == {carole.name}
