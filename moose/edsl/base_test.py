import numpy as np
from absl.testing import parameterized

from moose.computation import dtypes
from moose.computation import host as host_ops
from moose.computation import standard as standard_ops
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import TensorConstant
from moose.computation.standard import TensorType
from moose.computation.standard import UnknownType
from moose.edsl import base as edsl
from moose.edsl.tracer import trace
from moose.testing import run_test_computation

_MOOSE_DTYPES = [
    dtypes.float32,
    dtypes.float64,
    dtypes.int32,
    dtypes.int64,
    dtypes.uint32,
    dtypes.uint64,
]
_NUMPY_DTYPES = [
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    np.uint32,
    np.uint64,
]


class EdslTest(parameterized.TestCase):
    @parameterized.parameters(
        {"op": op, "OP": OP, "op_name": op_name}
        for (op, OP, op_name) in zip(
            [edsl.add, edsl.div, edsl.mul, edsl.sub],
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
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = op(
                edsl.constant(1.0, placement=player0),
                edsl.constant(1.0, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        binary_op = concrete_comp.operation(f"{op_name}_0")
        assert binary_op == OP(
            placement_name="player0",
            name=f"{op_name}_0",
            inputs={"lhs": "constant_0", "rhs": "constant_1"},
            output_type=TensorType(dtype=dtypes.float64),
        )

    def test_concatenate(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.concatenate(
                [
                    edsl.constant(np.array([1], dtype=np.float32), placement=player0),
                    edsl.constant(np.array([1], dtype=np.float32), placement=player0),
                ],
                axis=1,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("concatenate_0")
        assert op == standard_ops.ConcatenateOperation(
            placement_name="player0",
            name="concatenate_0",
            axis=1,
            inputs={"array0": "constant_0", "array1": "constant_1"},
            output_type=TensorType(dtype=dtypes.float32),
        )

    def test_ones(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            shape = edsl.constant([2, 2], placement=player0)
            x0 = edsl.ones(shape, dtype=dtypes.float64, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("ones_0")
        assert op == standard_ops.OnesOperation(
            placement_name="player0",
            name="ones_0",
            dtype=dtypes.float64,
            inputs={"shape": "constant_0"},
            output_type=TensorType(dtype=dtypes.float64),
        )

    def test_square(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.square(
                edsl.constant(np.array([1.0], dtype=np.float32), placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("mul_0")
        assert op == standard_ops.MulOperation(
            placement_name="player0",
            name="mul_0",
            inputs={"lhs": "constant_0", "rhs": "constant_0"},
            output_type=TensorType(dtype=dtypes.float32),
        )

    @parameterized.parameters(
        (edsl.sum, standard_ops.SumOperation, "sum", None),
        (edsl.sum, standard_ops.SumOperation, "sum", 0),
        (edsl.mean, standard_ops.MeanOperation, "mean", None),
        (edsl.mean, standard_ops.MeanOperation, "mean", 0),
    )
    def test_reduce_op(self, reduce_op_fn, reduce_op_cls, reduce_op_name, axis):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = reduce_op_fn(
                edsl.constant(np.array([1, 1], dtype=np.float32), placement=player0),
                axis=axis,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        concrete_op_name = "{}_0".format(reduce_op_name)
        op = concrete_comp.operation(concrete_op_name)
        assert op == reduce_op_cls(
            placement_name="player0",
            name=concrete_op_name,
            axis=axis,
            inputs={"x": "constant_0"},
            output_type=TensorType(dtype=dtypes.float32),
        )

    def test_transpose(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.transpose(
                edsl.constant(np.array([1], dtype=np.float32), placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("transpose_0")
        assert op == standard_ops.TransposeOperation(
            placement_name="player0",
            name="transpose_0",
            axes=None,
            inputs={"x": "constant_0"},
            output_type=TensorType(dtype=dtypes.float32),
        )

    @parameterized.parameters(
        (np.array([1.0, 2.0, 3.0]), (-1,), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), (-1, 1), np.array([[1.0], [2.0], [3.0]])),
        (np.array([1.0, 2.0, 3.0, 4.0]), (1, 4), np.array([[1.0, 2.0, 3.0, 4.0]])),
        (np.array([1.0, 2.0, 3.0, 4.0]), (2, 2), np.array([[1.0, 2.0], [3.0, 4.0]])),
    )
    def test_reshape(self, x, shape, expected):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            original = edsl.constant(x, placement=player0)
            actual = edsl.reshape(original, shape, placement=player0)
            return edsl.save("actual", actual, placement=player0)

        concrete_comp = trace(my_comp)

        results = run_test_computation(concrete_comp, [player0])
        np.testing.assert_equal(results[player0]["actual"], expected)

    @parameterized.parameters((np.array(1), np.array([[1]])),)
    def test_atleast_2d(self, x, expected):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            input = edsl.constant(np.array([1.0]), placement=player0)
            out = edsl.atleast_2d(input, to_column_vector=True, placement=player0)
            return out

        concrete_comp = trace(my_comp)

        op = concrete_comp.operation("atleast_2d_0")
        assert op == standard_ops.AtLeast2DOperation(
            placement_name="player0",
            name="atleast_2d_0",
            inputs={"x": "constant_0"},
            to_column_vector=True,
            output_type=TensorType(dtype=dtypes.float64),
        )

    @parameterized.parameters(None, 1)
    def test_squeeze(self, axis):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.squeeze(
                edsl.constant(np.array([[1]]), placement=player0),
                axis=axis,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("squeeze_0")
        assert op == standard_ops.SqueezeOperation(
            placement_name="player0",
            name="squeeze_0",
            inputs={"x": "constant_0"},
            axis=axis,
            output_type=TensorType(dtype=dtypes.int64),
        )

    def test_unsqueeze(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.expand_dims(
                edsl.constant(np.array([1.0]), placement=player0),
                axis=1,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("expand_dims_0")
        assert op == standard_ops.ExpandDimsOperation(
            placement_name="player0",
            name="expand_dims_0",
            inputs={"x": "constant_0"},
            axis=1,
            output_type=TensorType(dtype=dtypes.float64),
        )

    def test_constant(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.constant(1.0, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        constant_op = concrete_comp.operation("constant_0")
        assert constant_op == standard_ops.ConstantOperation(
            placement_name="player0",
            name="constant_0",
            inputs={},
            value=TensorConstant(value=[1.0]),
            output_type=TensorType(dtype=dtypes.float64),
        )

    def test_load(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.load(
                edsl.constant("stored_data", placement=player0),
                dtype=dtypes.float32,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        constant_op = concrete_comp.operation("load_0")
        assert constant_op == standard_ops.LoadOperation(
            placement_name="player0",
            name="load_0",
            inputs={"key": "constant_0", "query": "constant_1"},
            output_type=TensorType(dtype=dtypes.float32),
        )

    def test_load_with_query(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.load(
                edsl.constant("stored_data", placement=player0),
                query=edsl.constant('{"select_columns": ["x"]}', placement=player0),
                dtype=dtypes.float32,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        constant_op = concrete_comp.operation("load_0")
        assert constant_op == standard_ops.LoadOperation(
            placement_name="player0",
            name="load_0",
            inputs={"key": "constant_0", "query": "constant_1"},
            output_type=TensorType(dtype=dtypes.float32),
        )

    def test_arguments(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp(x: edsl.Argument(placement=player0, dtype=dtypes.float64)):
            y = edsl.constant(1.0, placement=player0)
            z = edsl.add(x, y, placement=player0)
            return z

        concrete_comp = trace(my_comp)

        assert concrete_comp == Computation(
            operations={
                "x": standard_ops.InputOperation(
                    placement_name="player0",
                    name="x",
                    inputs={},
                    output_type=TensorType(dtype=dtypes.float64),
                ),
                "constant_0": standard_ops.ConstantOperation(
                    placement_name="player0",
                    name="constant_0",
                    inputs={},
                    value=TensorConstant(value=[1.0]),
                    output_type=TensorType(dtype=dtypes.float64),
                ),
                "add_0": standard_ops.AddOperation(
                    placement_name="player0",
                    name="add_0",
                    inputs={"lhs": "x", "rhs": "constant_0"},
                    output_type=TensorType(dtype=dtypes.float64),
                ),
                "output_0": standard_ops.OutputOperation(
                    placement_name="player0",
                    name="output_0",
                    inputs={"value": "add_0"},
                    output_type=TensorType(dtype=dtypes.float64),
                ),
            },
            placements={"player0": HostPlacement(name="player0")},
        )

    def test_run_program(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.run_program(
                "python",
                ["local_computation.py"],
                edsl.constant(1, placement=player0),
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

    @parameterized.parameters(
        (1, None, dtypes.int64),
        (1, dtypes.int64, dtypes.int64),
        (1, dtypes.int32, dtypes.int32),
        (1.0, dtypes.float32, dtypes.float32),
        (1.0, dtypes.float64, dtypes.float64),
        (1.0, None, dtypes.float64),
        (np.array([1.0]), None, dtypes.float64),
        (np.array([1.0], dtype=np.float32), dtypes.float32, dtypes.float32),
        (np.array([1.0]), dtypes.float64, dtypes.float64),
    )
    def test_cast_noop(self, input_value, from_dtype, into_dtype):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            with player0:
                x = edsl.constant(input_value, dtype=from_dtype)
                x_new = edsl.cast(x, dtype=into_dtype)
                return x_new

        concrete_comp = trace(my_comp)
        self.assertRaises(KeyError, lambda: concrete_comp.operation("cast_0"))

    @parameterized.parameters(
        # Cast python native numbers
        *[
            (value, from_dtype, into_dtype)
            for value in (1, 1.0)
            for from_dtype in _MOOSE_DTYPES
            for into_dtype in _MOOSE_DTYPES
            if from_dtype != into_dtype
        ],
        # Cast numpy array, explicit moose dtype
        *[
            (np.array([1.0]), dtypes.float64, into_dtype)
            for into_dtype in _MOOSE_DTYPES
            if into_dtype != dtypes.float64
        ],
        *[
            (np.array([1]), dtypes.int64, into_dtype)
            for into_dtype in _MOOSE_DTYPES
            if into_dtype != dtypes.int64
        ],
        # Cast numpy array w/ implicit dtype
        *[
            (np.array([1], dtype=from_dtype), None, into_dtype)
            for from_dtype in _NUMPY_DTYPES
            for into_dtype in _MOOSE_DTYPES
            if into_dtype.numpy_dtype != from_dtype
        ],
    )
    def test_cast(self, input_value, from_dtype, into_dtype):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            with player0:
                x = edsl.constant(input_value, dtype=from_dtype)
                x_new = edsl.cast(x, dtype=into_dtype)
                return x_new

        concrete_comp = trace(my_comp)
        cast_op = concrete_comp.operation("cast_0")
        assert cast_op == standard_ops.CastOperation(
            placement_name="player0",
            name="cast_0",
            inputs={"x": "constant_0"},
            output_type=TensorType(dtype=into_dtype),
        )

    @parameterized.parameters(
        *[
            (np.array([1.0]), into_dtype)
            for into_dtype in _MOOSE_DTYPES
            if into_dtype != dtypes.float64
        ],
        *[
            (np.array([1]), into_dtype)
            for into_dtype in _MOOSE_DTYPES
            if into_dtype != dtypes.int64
        ],
    )
    def test_implicit_cast(self, input_value, dtype):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            with player0:
                x = edsl.constant(input_value, dtype=dtype)
                return x

        concrete_comp = trace(my_comp)
        cast_op = concrete_comp.operation("cast_0")
        assert cast_op == standard_ops.CastOperation(
            placement_name="player0",
            name="cast_0",
            inputs={"x": "constant_0"},
            output_type=TensorType(dtype=dtype),
        )


def _npdtype_into_moose_dtype(npdtype):
    return dtypes._NUMPY_DTYPES_MAP(npdtype)
