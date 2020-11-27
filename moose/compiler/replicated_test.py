import dill
from absl.testing import parameterized

from moose.compiler.computation import AddOperation
from moose.compiler.computation import CallPythonFunctionOperation
from moose.compiler.computation import ConstantOperation
from moose.compiler.computation import DivOperation
from moose.compiler.computation import MulOperation
from moose.compiler.computation import ReceiveOperation
from moose.compiler.computation import RunProgramOperation
from moose.compiler.computation import SendOperation
from moose.compiler.computation import SubOperation
from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import div
from moose.compiler.edsl import function
from moose.compiler.edsl import mul
from moose.compiler.edsl import run_program
from moose.compiler.edsl import sub
from moose.compiler.replicated import ReplicatedPlacement


class ReplicatedTest(parameterized.TestCase):
    def test_replicated(self):
        alice = HostPlacement(name="alice")
        bob = HostPlacement(name="bob")
        carole = HostPlacement(name="carole")
        replicated = ReplicatedPlacement("replicated", alice, bob, carole)
        dave = HostPlacement(name="dave")

        @computation
        def my_comp():
            x = constant(3, placement=alice)
            y = constant(4, placement=bob)
            z = mul(x, y, placement=replicated)
            v = constant(1, placement=dave)
            w = add(z, v, placement=dave)
            return w

        concrete_comp = my_comp.trace_func(render=True)
        concrete_comp.render()

        send_ops = [
            op for op in concrete_comp.operations() if isinstance(op, SendOperation)
        ]
        assert len(send_ops) == 4, [f"{op.sender} -> {op.receiver}" for op in send_ops]

        recv_ops = [
            op for op in concrete_comp.operations() if isinstance(op, ReceiveOperation)
        ]
        assert len(recv_ops) == 4, [f"{op.sender} -> {op.receiver}" for op in recv_ops]
