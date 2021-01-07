import dill
import numpy as np
from absl.testing import parameterized

from moose.computation import host as host_ops
from moose.computation import standard as standard_ops
from moose.computation.base import Computation
from moose.computation.base import UnknownType
from moose.computation.host import HostPlacement
from moose.computation.standard import TensorType
from moose.edsl.base import Argument
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import div
from moose.edsl.base import function
from moose.edsl.base import host_placement
from moose.edsl.base import mul
from moose.edsl.base import ones
from moose.edsl.base import run_program
from moose.edsl.base import sub
from moose.edsl.base import transpose
from moose.edsl.tracer import trace


class EdslTest(parameterized.TestCase):
    @parameterized.parameters(
        {"op": op, "OP": OP, "op_name": op_name}
        for (op, OP, op_name) in zip(
            [add, div, mul, sub],
            [
                standard_ops.AddOperation,
                standard_ops.DivOperation,
                standard_ops.MulOperation,
                standard_ops.SubOperation,
            ],
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
            output_type=TensorType(datatype="float"),
        )

    def test_ones(self):
        player0 = host_placement(name="player0")

        @computation
        def my_comp():
            x0 = ones((2, 2), dtype=np.float64, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("ones_0")
        assert op == standard_ops.OnesOperation(
            placement_name="player0",
            name="ones_0",
            shape=(2, 2),
            dtype=np.float64,
            inputs={},
            output_type=TensorType(datatype="float"),
        )

    def test_transpose(self):
        player0 = host_placement(name="player0")

        @computation
        def my_comp():
            x0 = transpose(
                constant(np.array([1]), placement=player0), placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("transpose_0")
        assert op == standard_ops.TransposeOperation(
            placement_name="player0",
            name="transpose_0",
            axes=None,
            inputs={"x": "constant_0"},
            output_type=TensorType(datatype="float"),
        )

    def test_call_python_fn(self):
        player0 = host_placement(name="player0")

        @function(output_type=float)
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
        assert call_py_op == host_ops.CallPythonFunctionOperation(
            placement_name="player0",
            name="call_python_function_0",
            inputs={"arg0": "constant_0"},
            pickled_fn=pickled_fn,
            output_type=TensorType(datatype="float"),
        )

    def test_constant(self):
        player0 = host_placement(name="player0")

        @computation
        def my_comp():
            x0 = constant(1, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        constant_op = concrete_comp.operation("constant_0")
        assert constant_op == standard_ops.ConstantOperation(
            placement_name="player0",
            name="constant_0",
            inputs={},
            value=1,
            output_type=TensorType(datatype="float"),
        )

    def test_arguments(self):
        player0 = host_placement(name="player0")

        @computation
        def my_comp(x: Argument(placement=player0, datatype=float)):
            y = constant(1, placement=player0)
            z = add(x, y, placement=player0)
            return z

        concrete_comp = trace(my_comp)

        assert concrete_comp == Computation(
            operations={
                "x": standard_ops.InputOperation(
                    placement_name="player0",
                    name="x",
                    inputs={},
                    output_type=TensorType(datatype="float"),
                ),
                "constant_0": standard_ops.ConstantOperation(
                    placement_name="player0",
                    name="constant_0",
                    inputs={},
                    value=1,
                    output_type=TensorType(datatype="float"),
                ),
                "add_0": standard_ops.AddOperation(
                    placement_name="player0",
                    name="add_0",
                    inputs={"lhs": "x", "rhs": "constant_0"},
                    output_type=TensorType(datatype="float"),
                ),
                "output_0": standard_ops.OutputOperation(
                    placement_name="player0",
                    name="output_0",
                    inputs={"value": "add_0"},
                ),
            },
            placements={"player0": HostPlacement(name="player0")},
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

        assert script_py_op == host_ops.RunProgramOperation(
            placement_name="player0",
            name="run_program_0",
            inputs={"arg0": "constant_0"},
            path="python",
            args=["local_computation.py"],
            output_type=UnknownType(),
        )
