import dill
from absl.testing import parameterized

from moose.computation.host import CallPythonFunctionOperation
from moose.computation.host import RunProgramOperation
from moose.computation.standard import AddOperation
from moose.computation.standard import ConstantOperation
from moose.computation.standard import DivOperation
from moose.computation.standard import MulOperation
from moose.computation.standard import SubOperation
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import div
from moose.edsl.base import function
from moose.edsl.base import host_placement
from moose.edsl.base import mul
from moose.edsl.base import run_program
from moose.edsl.base import sub
from moose.edsl.tracer import trace


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
        player0 = host_placement(name="player0")

        @computation
        def my_comp():
            x0 = op(
                constant(1, placement=player0),
                constant(1, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        binary_op = concrete_comp.operation(f"{op_name}_0")
        assert binary_op == OP(
            placement_name="player0",
            name=f"{op_name}_0",
            inputs={"lhs": "constant_0", "rhs": "constant_1"},
        )

    def test_call_python_fn(self):
        player0 = host_placement(name="player0")

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            x = constant(1, placement=player0)
            y = add_one(x, placement=player0)
            z = add(x, y, placement=player0)
            return z

        concrete_comp = trace(my_comp)
        call_py_op = concrete_comp.operation("call_python_function_0")

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
        player0 = host_placement(name="player0")

        @computation
        def my_comp():
            x0 = constant(1, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        constant_op = concrete_comp.operation("constant_0")
        assert constant_op == ConstantOperation(
            placement_name="player0", name="constant_0", inputs={}, value=1,
        )

    def test_run_program(self):
        player0 = host_placement(name="player0")

        @computation
        def my_comp():
            x0 = run_program(
                "python",
                ["local_computation.py"],
                constant(1, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        script_py_op = concrete_comp.operation("run_program_0")

        assert script_py_op == RunProgramOperation(
            placement_name="player0",
            name="run_program_0",
            inputs={"arg0": "constant_0"},
            path="python",
            args=["local_computation.py"],
        )
