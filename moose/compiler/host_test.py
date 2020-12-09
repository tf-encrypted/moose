from absl.testing import parameterized

from moose.computation.standard import ReceiveOperation
from moose.computation.standard import SendOperation
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import host_placement
from moose.edsl.base import mul
from moose.edsl.tracer import trace


class HostTest(parameterized.TestCase):
    def test_send_receive(self):
        player0 = host_placement(name="player0")
        player1 = host_placement(name="player1")

        @computation
        def my_comp():
            x0 = constant(1, placement=player0)
            x1 = constant(1, placement=player0)
            x2 = add(x0, x1, placement=player1)

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
            output_type_name=None,  # TODO
        )

    def test_pass_networking(self):
        alice = host_placement(name="alice")
        bob = host_placement(name="bob")
        carole = host_placement(name="carole")
        dave = host_placement(name="dave")

        @computation
        def my_comp():
            a = constant(1, placement=alice)
            b = constant(2, placement=bob)
            c1 = add(a, b, placement=carole)
            c2 = add(a, b, placement=carole)
            c3 = mul(c1, c2, placement=carole)
            d = add(a, c3, placement=dave)
            return d

        concrete_comp = trace(my_comp)

        send_ops = concrete_comp.find_operations_of_type(SendOperation)
        assert len(send_ops) == 4, [f"{op.sender} -> {op.receiver}" for op in send_ops]

        recv_ops = concrete_comp.find_operations_of_type(ReceiveOperation)
        assert len(recv_ops) == 4, [f"{op.sender} -> {op.receiver}" for op in recv_ops]
