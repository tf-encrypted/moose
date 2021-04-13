import numpy as np
from absl.testing import parameterized

from moose.computation import dtypes
from moose.computation import fixedpoint as fixedpoint_ops
from moose.computation import ring as ring_ops
from moose.computation import standard as standard_ops
from moose.computation.standard import IntType
from moose.computation.standard import ReceiveOperation
from moose.computation.standard import SendOperation
from moose.edsl import base as edsl
from moose.edsl.tracer import trace


class HostTest(parameterized.TestCase):
    def test_fixedpoint_example_lowering(self):
        alice = edsl.host_placement(name="alice")

        @edsl.computation
        def my_comp():
            x = edsl.constant(
                np.array([10.0, 12.0]), dtype=dtypes.fixed(8, 27), placement=alice
            )
            y = edsl.mul(x, x, placement=alice)
            z = edsl.cast(y, dtype=dtypes.float64, placement=alice)
            return z

        concrete_comp = trace(my_comp)
        fp_comp = concrete_comp.find_operations_of_type(
            fixedpoint_ops.FixedpointOperation
        )
        encode = concrete_comp.find_operations_of_type(
            fixedpoint_ops.RingEncodeOperation
        )
        decode = concrete_comp.find_operations_of_type(
            fixedpoint_ops.RingEncodeOperation
        )
        ring_comp = concrete_comp.find_operations_of_type(ring_ops.RingOperation)
        ring_mul = concrete_comp.find_operations_of_type(ring_ops.RingMulOperation)
        ring_trunc = concrete_comp.find_operations_of_type(ring_ops.RingShrOperation)
        const = concrete_comp.find_operations_of_type(standard_ops.ConstantOperation)
        output = concrete_comp.find_operations_of_type(standard_ops.OutputOperation)

        assert len(concrete_comp.operations) == 6
        assert len(fp_comp) == 2
        assert len(ring_comp) == 2
        assert len(encode) == 1
        assert len(decode) == 1
        assert len(ring_mul) == 1
        assert len(ring_trunc) == 1
        assert len(const) == 1
        assert len(output) == 1

    def test_send_receive(self):
        player0 = edsl.host_placement(name="player0")
        player1 = edsl.host_placement(name="player1")

        @edsl.computation
        def my_comp():
            x0 = edsl.constant(1, placement=player0)
            x1 = edsl.constant(1, placement=player0)
            x2 = edsl.add(x0, x1, placement=player1)

            return x2

        concrete_comp = trace(my_comp)

        send_op = concrete_comp.operation("send_0")
        assert send_op == SendOperation(
            placement_name="player0",
            name="send_0",
            inputs={"value": "serialize_0"},
            sender="player0",
            receiver="player1",
            rendezvous_key="rendezvous_key_0",
        )
        receive_op = concrete_comp.operation("receive_0")
        assert receive_op == ReceiveOperation(
            placement_name="player1",
            name="receive_0",
            inputs={},
            sender="player0",
            receiver="player1",
            rendezvous_key="rendezvous_key_0",
            output_type=IntType(),
        )

    def test_pass_networking(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        dave = edsl.host_placement(name="dave")

        @edsl.computation
        def my_comp():
            a = edsl.constant(1, placement=alice)
            b = edsl.constant(2, placement=bob)
            c1 = edsl.add(a, b, placement=carole)
            c2 = edsl.add(a, b, placement=carole)
            c3 = edsl.mul(c1, c2, placement=carole)
            d = edsl.add(a, c3, placement=dave)
            return d

        concrete_comp = trace(my_comp)

        send_ops = concrete_comp.find_operations_of_type(SendOperation)
        assert len(send_ops) == 4, [f"{op.sender} -> {op.receiver}" for op in send_ops]

        recv_ops = concrete_comp.find_operations_of_type(ReceiveOperation)
        assert len(recv_ops) == 4, [f"{op.sender} -> {op.receiver}" for op in recv_ops]
