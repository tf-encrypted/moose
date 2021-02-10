from absl.testing import parameterized

from moose.computations.standard import ReceiveOperation
from moose.computations.standard import SendOperation
from moose.edsl import base as edsl
from moose.edsl.tracer import trace


class HostTest(parameterized.TestCase):
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
