from absl.testing import parameterized

from moose.computation.mpspdz import MpspdzCallOperation
from moose.computation.mpspdz import MpspdzLoadOutputOperation
from moose.computation.mpspdz import MpspdzSaveInputOperation
from moose.computation.standard import TensorType
from moose.edsl import base as edsl
from moose.edsl import dtypes
from moose.edsl.tracer import trace


class MpspdzTest(parameterized.TestCase):
    def test_mpspdz(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        mpspdz = edsl.mpspdz_placement("mpspdz", players=[alice, bob, carole])

        @edsl.function(output_type=TensorType(dtypes.int64))
        def my_function(x, y, z):
            return x * y + z

        @edsl.computation
        def my_comp():
            x = edsl.constant(1, placement=alice)
            y = edsl.constant(2, placement=bob)
            z = edsl.constant(3, placement=alice)
            v = my_function(x, y, z, placement=mpspdz)
            w = edsl.mul(v, v, placement=carole)
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
