import dill
from absl.testing import parameterized

from computation import AddOperation
from computation import CallPythonFunctionOperation
from computation import ConstantOperation
from computation import DivOperation
from computation import MulOperation
from computation import ReceiveOperation
from computation import RunPythonScriptOperation
from computation import SendOperation
from computation import SubOperation
from edsl import Role
from edsl import add
from edsl import computation
from edsl import constant
from edsl import div
from edsl import function
from edsl import mul
from edsl import run_python_script
from edsl import sub


class EdslTest(parameterized.TestCase):
    def test_constant(self):
        player0 = Role(name="player0")

        @computation
        def my_comp():
            with player0:
                x0 = constant(1)
            return x0

        concrete_comp = my_comp.trace_func()
        constant_op = concrete_comp.graph.nodes["constant_op0"]
        assert constant_op == ConstantOperation(
            device_name="player0",
            name="constant_op0",
            inputs={},
            output="constant0",
            value=1,
        )

    @parameterized.parameters(
        {"op": op, "OP": OP, "op_name": op_name}
        for (op, OP, op_name) in zip(
            [add, div, mul, sub],
            [AddOperation, DivOperation, MulOperation, SubOperation],
            ["add", "div", "mul", "sub"],
        )
    )
    def test_binary_op(self, op, OP, op_name):
        player0 = Role(name="player0")

        @computation
        def my_comp():
            with player0:
                x0 = op(constant(1), constant(1))
            return x0

        concrete_comp = my_comp.trace_func()
        binary_op = concrete_comp.graph.nodes[f"{op_name}operation_op0"]
        assert binary_op == OP(
            device_name="player0",
            name=f"{op_name}operation_op0",
            inputs={"lhs": "constant0", "rhs": "constant1"},
            output=f"{op_name}operation0",
        )

    def test_send_receive(self):
        player0 = Role(name="player0")
        player1 = Role(name="player1")

        @computation
        def my_comp():
            with player0:
                x0 = constant(1)
            with player1:
                x1 = add(x0, x0)

            return x1

        concrete_comp = my_comp.trace_func()

        send_op = concrete_comp.graph.nodes["send_op0"]
        assert send_op == SendOperation(
            device_name="player0",
            name="send_op0",
            inputs={"value": "constant0"},
            output=None,
            sender="player0",
            receiver="player1",
            rendezvous_key="rendezvous_key0",
        )
        receive_op = concrete_comp.graph.nodes["receive_op0"]
        assert receive_op == ReceiveOperation(
            device_name="player1",
            name="receive_op0",
            inputs={},
            output="receive0",
            sender="player0",
            receiver="player1",
            rendezvous_key="rendezvous_key0",
        )

    def test_run_python_script(self):
        player0 = Role(name="player0")

        @computation
        def my_comp():
            with player0:
                x0 = run_python_script("local_computation.py", constant(1))
            return x0

        concrete_comp = my_comp.trace_func()
        script_op = concrete_comp.graph.nodes["run_python_script_op0"]

        assert script_op == RunPythonScriptOperation(
            device_name="player0",
            name="run_python_script_op0",
            inputs={"arg0": "constant0"},
            output="run_python_script0",
            path="local_computation.py",
        )

    def test_call_python_fn(self):
        player0 = Role(name="player0")

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            with player0:
                x0 = add_one(constant(1))
            return x0

        concrete_comp = my_comp.trace_func()
        call_py_op = concrete_comp.graph.nodes["call_python_function_op0"]

        call_py_op.fn = dill.dumps(add_one)
        assert call_py_op == CallPythonFunctionOperation(
            device_name="player0",
            name="call_python_function_op0",
            inputs={"arg0": "constant0"},
            output="call_python_function0",
            fn=dill.dumps(add_one),
        )
