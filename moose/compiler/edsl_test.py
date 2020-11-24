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


class EdslTest(parameterized.TestCase):
    @parameterized.parameters(
        {"op": op, "OP": OP, "op_name": op_name}
        for (op, OP, op_name) in zip(
            [add, div, mul, sub],
            [AddOperation, DivOperation, MulOperation, SubOperation],
            ["add", "div", "mul", "sub"],
        )
    )
    def test_binary_op(self, op, OP, op_name):
        player0 = HostPlacement(name="player0")

        @computation
        def my_comp():
            x0 = op(
                constant(1, placement=player0),
                constant(1, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = my_comp.trace_func()
        binary_op = concrete_comp.graph.nodes[f"{op_name}operation_0"]
        assert binary_op == OP(
            placement_name="player0",
            name=f"{op_name}operation_0",
            inputs={"lhs": "constant_0", "rhs": "constant_1"},
        )

    def test_call_python_fn(self):
        player0 = HostPlacement(name="player0")

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            x0 = add_one(constant(1, placement=player0), placement=player0)
            return x0

        concrete_comp = my_comp.trace_func()
        call_py_op = concrete_comp.graph.nodes["call_python_function_0"]

        # TODO(Morten) for some reason the pickled functions deviated;
        # figure out why and improve test
        pickled_fn = dill.dumps(add_one)
        call_py_op.pickled_fn = pickled_fn
        assert call_py_op == CallPythonFunctionOperation(
            placement_name="player0",
            name="call_python_function_0",
            inputs={"arg0": "constant_0"},
            pickled_fn=pickled_fn,
            output_type=None,
        )

    def test_constant(self):
        player0 = HostPlacement(name="player0")

        @computation
        def my_comp():
            x0 = constant(1, placement=player0)
            return x0

        concrete_comp = my_comp.trace_func()
        constant_op = concrete_comp.graph.nodes["constant_0"]
        assert constant_op == ConstantOperation(
            placement_name="player0", name="constant_0", inputs={}, value=1,
        )

    def test_send_receive(self):
        player0 = HostPlacement(name="player0")
        player1 = HostPlacement(name="player1")

        @computation
        def my_comp():
            x0 = constant(1, placement=player0)
            x1 = add(x0, x0, placement=player1)

            return x1

        concrete_comp = my_comp.trace_func()

        send_op = concrete_comp.graph.nodes["send_0"]
        assert send_op == SendOperation(
            placement_name="player0",
            name="send_0",
            inputs={"value": "serialize_0"},
            sender="player0",
            receiver="player1",
            rendezvous_key="rendezvous_key_0",
        )
        receive_op = concrete_comp.graph.nodes["receive_0"]
        assert receive_op == ReceiveOperation(
            placement_name="player1",
            name="receive_0",
            inputs={},
            sender="player0",
            receiver="player1",
            rendezvous_key="rendezvous_key_0",
        )

    def test_run_program(self):
        player0 = HostPlacement(name="player0")

        @computation
        def my_comp():
            x0 = run_program(
                "python",
                ["local_computation.py"],
                constant(1, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = my_comp.trace_func()
        script_py_op = concrete_comp.graph.nodes["run_program_0"]

        assert script_py_op == RunProgramOperation(
            placement_name="player0",
            name="run_program_0",
            inputs={"arg0": "constant_0"},
            path="python",
            args=["local_computation.py"],
        )
