import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose.computation import computation as comp
from pymoose.computation import dtypes
from pymoose.computation import operations as ops
from pymoose.computation import placements as plc
from pymoose.computation import types as ty
from pymoose.computation import values
from pymoose.edsl import base as edsl
from pymoose.edsl.tracer import trace

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


def _iadd(x, y):
    x += y
    return x


def _isub(x, y):
    x -= y
    return x


def _imul(x, y):
    x *= y
    return x


def _idiv(x, y):
    x /= y
    return x


def _imatmul(x, y):
    x @= y
    return x


_DUNDER_METHODS = {
    "__add__": lambda x, y: x + y,
    "__sub__": lambda x, y: x - y,
    "__mul__": lambda x, y: x * y,
    "__truediv__": lambda x, y: x / y,
    "__matmul__": lambda x, y: x @ y,
    "__iadd__": _iadd,
    "__isub__": _isub,
    "__imul__": _imul,
    "__itruediv__": _idiv,
    "__imatmul__": _imatmul,
    "__gt__": lambda x, y: x > y,
    "__lt__": lambda x, y: x < y,
}


def build_ez_swap():
    alice = edsl.host_placement("alice")
    bob = edsl.host_placement("bob")
    carole = edsl.host_placement("carole")

    @edsl.computation
    def ez_swap(x: edsl.Argument(alice, dtype=dtypes.float64)):
        with bob:
            y = edsl.add(x, x)

        with carole:
            z = edsl.mul(y, y)

        return z

    return ez_swap


def build_mirr_swap():
    alice = edsl.host_placement("alice")
    bob = edsl.host_placement("bob")
    carole = edsl.host_placement("carole")
    mirr = edsl.mirrored_placement("mirrored", [alice, bob, carole])
    fp_dtype = dtypes.fixed(14, 23)

    @edsl.computation
    def mirr_swap(x: edsl.Argument(alice, dtype=fp_dtype)):
        with mirr:
            y = edsl.add(x, x)

        with carole:
            z = edsl.mul(y, y)

        return z

    return mirr_swap


def build_repl_swap():
    alice = edsl.host_placement("alice")
    bob = edsl.host_placement("bob")
    carole = edsl.host_placement("carole")
    repl = edsl.replicated_placement("replicated", [alice, bob, carole])
    fp_dtype = dtypes.fixed(14, 23)

    @edsl.computation
    def repl_swap(x: edsl.Argument(alice, dtype=fp_dtype)):
        with repl:
            y = edsl.add(x, x)

        with carole:
            z = edsl.mul(y, y)

        return z

    return repl_swap


class EdslTest(parameterized.TestCase):
    @parameterized.parameters(
        ("__add__", ops.AddOperation, "add"),
        ("__sub__", ops.SubOperation, "sub"),
        ("__mul__", ops.MulOperation, "mul"),
        ("__truediv__", ops.DivOperation, "div"),
        ("__matmul__", ops.DotOperation, "dot"),
        ("__iadd__", ops.AddOperation, "add"),
        ("__isub__", ops.SubOperation, "sub"),
        ("__imul__", ops.MulOperation, "mul"),
        ("__itruediv__", ops.DivOperation, "div"),
        ("__imatmul__", ops.DotOperation, "dot"),
        ("__gt__", ops.GreaterOperation, "greater"),
        ("__lt__", ops.LessOperation, "less"),
    )
    def test_binary_dunder_methods(self, dunder_name, op_cls, op_name):
        alice = edsl.host_placement("alice")
        if op_name in ["greater", "less"]:
            expected_output_dtype = dtypes.bool_
        else:
            expected_output_dtype = dtypes.float64

        @edsl.computation
        def dunder_comp():
            with alice:
                x = edsl.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
                y = _DUNDER_METHODS[dunder_name](x, x)
            return y

        traced_comp = trace(dunder_comp)
        binary_op = traced_comp.operation(f"{op_name}_0")
        assert binary_op == op_cls(
            placement_name="alice",
            name=f"{op_name}_0",
            inputs={"lhs": "constant_0", "rhs": "constant_0"},
            signature=ops.OpSignature(
                {
                    "lhs": ty.TensorType(dtypes.float64),
                    "rhs": ty.TensorType(dtypes.float64),
                },
                ty.TensorType(expected_output_dtype),
            ),
        )

    def test_dunder_abs(self):
        alice = edsl.host_placement("alice")

        @edsl.computation
        def dunder_comp():
            with alice:
                x = edsl.constant(np.array([1.0, -2.0, 3.0], dtype=np.float64))
                y = abs(x)
            return y

        traced_comp = trace(dunder_comp)
        abs_op = traced_comp.operation("abs_0")
        assert abs_op == ops.AbsOperation(
            placement_name="alice",
            name="abs_0",
            inputs={"x": "constant_0"},
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.float64)},
                ty.TensorType(dtypes.float64),
            ),
        )

    @parameterized.parameters(
        dtypes.float64,
        dtypes.fixed(14, 23),
    )
    def test_dunder_neg(self, dtype):
        alice = edsl.host_placement("alice")

        @edsl.computation
        def dunder_comp():
            with alice:
                x = edsl.constant(np.array([1.0, -2.0, 3.0]), dtype=dtype)
                y = -x
            return y

        traced_comp = trace(dunder_comp)
        neg_mul_op = traced_comp.operation("mul_0")
        if dtype.is_fixedpoint:
            lhs_name, rhs_name = "cast_0", "cast_1"
        else:
            lhs_name, rhs_name = "constant_0", "constant_1"
        assert neg_mul_op == ops.MulOperation(
            placement_name="alice",
            name="mul_0",
            # "constant_0" is the implicit -1 constant, and "constant_1" is the x array
            inputs={"lhs": lhs_name, "rhs": rhs_name},
            signature=ops.OpSignature(
                {
                    "lhs": ty.TensorType(dtype),
                    "rhs": ty.TensorType(dtype),
                },
                ty.TensorType(dtype),
            ),
        )

    def test_identity(self):
        alice = edsl.host_placement("alice")
        bob = edsl.host_placement("bob")

        @edsl.computation
        def my_comp():
            with alice:
                c = edsl.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))

            with bob:
                c = edsl.identity(c)

            return c

        logical_comp = trace(my_comp)
        identity_op = logical_comp.operation("identity_0")
        assert identity_op == ops.IdentityOperation(
            placement_name="bob",
            name="identity_0",
            inputs={"x": "constant_0"},
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.float64)},
                ty.TensorType(dtypes.float64),
            ),
        )

    @parameterized.parameters(
        {"op": op, "OP": OP, "op_name": op_name}
        for (op, OP, op_name) in zip(
            [edsl.add, edsl.div, edsl.mul, edsl.sub],
            [ops.AddOperation, ops.DivOperation, ops.MulOperation, ops.SubOperation],
            ["add", "div", "mul", "sub"],
        )
    )
    def test_binary_op(self, op, OP, op_name):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = op(
                edsl.constant(1.0, dtype=dtypes.float64, placement=player0),
                edsl.constant(1.0, dtype=dtypes.float64, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        binary_op = concrete_comp.operation(f"{op_name}_0")
        assert binary_op == OP(
            placement_name="player0",
            name=f"{op_name}_0",
            inputs={"lhs": "constant_0", "rhs": "constant_1"},
            signature=ops.OpSignature(
                {
                    "lhs": ty.TensorType(dtypes.float64),
                    "rhs": ty.TensorType(dtypes.float64),
                },
                ty.TensorType(dtypes.float64),
            ),
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
        assert op == ops.ConcatenateOperation(
            placement_name="player0",
            name="concatenate_0",
            axis=1,
            inputs={"array0": "constant_0", "array1": "constant_1"},
            signature=ops.OpSignature(
                {
                    "array0": ty.TensorType(dtypes.float32),
                    "array1": ty.TensorType(dtypes.float32),
                },
                ty.TensorType(dtypes.float32),
            ),
        )

    def test_add_n(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.add_n(
                [
                    edsl.constant(np.array([1], dtype=np.float32), placement=player0),
                    edsl.constant(np.array([1], dtype=np.float32), placement=player0),
                ],
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("add_n_0")
        assert op == ops.AddNOperation(
            placement_name="player0",
            name="add_n_0",
            inputs={"array0": "constant_0", "array1": "constant_1"},
            signature=ops.OpSignature(
                {
                    "array0": ty.TensorType(dtypes.float32),
                    "array1": ty.TensorType(dtypes.float32),
                },
                ty.TensorType(dtypes.float32),
            ),
        )

    def test_ones(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            shape = edsl.constant([2, 2], vtype=ty.ShapeType(), placement=player0)
            x0 = edsl.ones(shape, dtype=dtypes.float64, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("ones_0")
        assert op == ops.OnesOperation(
            placement_name="player0",
            name="ones_0",
            inputs={"shape": "constant_0"},
            signature=ops.OpSignature(
                input_types={"shape": ty.ShapeType()},
                return_type=ty.TensorType(dtype=dtypes.float64),
            ),
        )

    def test_zeros(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            shape = edsl.constant([2, 2], vtype=ty.ShapeType(), placement=player0)
            x0 = edsl.zeros(shape, dtype=dtypes.float64, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("zeros_0")
        assert op == ops.ZerosOperation(
            placement_name="player0",
            name="zeros_0",
            inputs={"shape": "constant_0"},
            signature=ops.OpSignature(
                input_types={"shape": ty.ShapeType()},
                return_type=ty.TensorType(dtype=dtypes.float64),
            ),
        )

    def test_relu(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x = np.array([1.0, -2.0, 3.0, -4.0])
            x0 = edsl.relu(
                edsl.constant(x, dtype=dtypes.float64, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("relu_0")
        assert op == ops.ReluOperation(
            placement_name="player0",
            name="relu_0",
            inputs={"x": "constant_0"},
            signature=ops.OpSignature(
                input_types={"x": ty.TensorType(dtype=dtypes.float64)},
                return_type=ty.TensorType(dtype=dtypes.float64),
            ),
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
        assert op == ops.MulOperation(
            placement_name="player0",
            name="mul_0",
            inputs={"lhs": "constant_0", "rhs": "constant_0"},
            signature=ops.OpSignature(
                {
                    "lhs": ty.TensorType(dtypes.float32),
                    "rhs": ty.TensorType(dtypes.float32),
                },
                ty.TensorType(dtypes.float32),
            ),
        )

    @parameterized.parameters(
        (edsl.sum, ops.SumOperation, "sum", None),
        (edsl.sum, ops.SumOperation, "sum", 0),
        (edsl.mean, ops.MeanOperation, "mean", None),
        (edsl.mean, ops.MeanOperation, "mean", 0),
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
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.float32)},
                ty.TensorType(dtypes.float32),
            ),
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
        assert op == ops.TransposeOperation(
            placement_name="player0",
            name="transpose_0",
            inputs={"x": "constant_0"},
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.float32)},
                ty.TensorType(dtypes.float32),
            ),
        )

    def test_reshape(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        new_shape = (2, 2)
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            original = edsl.constant(x, placement=player0)
            actual = edsl.reshape(original, new_shape, placement=player0)
            return actual

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("reshape_0")
        assert op == ops.ReshapeOperation(
            placement_name="player0",
            name="reshape_0",
            inputs={"x": "constant_0", "shape": "constant_1"},
            signature=ops.OpSignature(
                input_types={
                    "x": ty.TensorType(dtype=dtypes.float64),
                    "shape": ty.ShapeType(),
                },
                return_type=ty.TensorType(dtype=dtypes.float64),
            ),
        )

    @parameterized.parameters(
        (np.array(1), np.array([[1]])),
    )
    def test_atleast_2d(self, x, expected):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            input = edsl.constant(np.array([1.0]), placement=player0)
            out = edsl.atleast_2d(input, to_column_vector=True, placement=player0)
            return out

        concrete_comp = trace(my_comp)

        op = concrete_comp.operation("atleast_2d_0")
        assert op == ops.AtLeast2DOperation(
            placement_name="player0",
            name="atleast_2d_0",
            inputs={"x": "constant_0"},
            to_column_vector=True,
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.float64)},
                ty.TensorType(dtypes.float64),
            ),
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
        assert op == ops.SqueezeOperation(
            placement_name="player0",
            name="squeeze_0",
            inputs={"x": "constant_0"},
            axis=axis,
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.int64)},
                ty.TensorType(dtypes.int64),
            ),
        )

    def test_index_axis(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.index_axis(
                edsl.constant(np.array([1.0]), placement=player0),
                axis=1,
                index=0,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("index_axis_0")
        assert op == ops.IndexAxisOperation(
            placement_name="player0",
            name="index_axis_0",
            inputs={"x": "constant_0"},
            axis=1,
            index=0,
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.float64)},
                ty.TensorType(dtypes.float64),
            ),
        )

    @parameterized.parameters({"axis": axis} for axis in [0, [0, 1]])
    def test_unsqueeze(self, axis):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.expand_dims(
                edsl.constant(np.array([1.0]), placement=player0),
                axis=axis,
                placement=player0,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("expand_dims_0")
        if isinstance(axis, int):
            axis = [axis]
        assert op == ops.ExpandDimsOperation(
            placement_name="player0",
            name="expand_dims_0",
            inputs={"x": "constant_0"},
            axis=axis,
            signature=ops.OpSignature(
                {"x": ty.TensorType(dtypes.float64)},
                ty.TensorType(dtypes.float64),
            ),
        )

    def test_mux(self):
        player0 = edsl.host_placement(name="player0")
        player1 = edsl.host_placement(name="player1")
        player2 = edsl.host_placement(name="player2")
        replicated = edsl.replicated_placement(
            name="replicated", players=[player0, player1, player2]
        )

        @edsl.computation
        def my_comp():
            x0 = edsl.mux(
                edsl.constant(np.array([True]), placement=player0),
                edsl.constant(
                    np.array([1.0]), placement=player0, dtype=dtypes.fixed(8, 27)
                ),
                edsl.constant(
                    np.array([0.0]), placement=player0, dtype=dtypes.fixed(8, 27)
                ),
                placement=replicated,
            )
            return x0

        concrete_comp = trace(my_comp)
        op = concrete_comp.operation("mux_0")
        assert op == ops.MuxOperation(
            placement_name="replicated",
            name="mux_0",
            inputs={"selector": "constant_0", "x": "cast_0", "y": "cast_1"},
            signature=ops.OpSignature(
                {
                    "selector": ty.TensorType(dtypes.bool_),
                    "x": ty.TensorType(dtypes.fixed(8, 27)),
                    "y": ty.TensorType(dtypes.fixed(8, 27)),
                },
                ty.TensorType(dtypes.fixed(8, 27)),
            ),
        )

    def test_constant(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = edsl.constant(1.0, dtype=dtypes.float64, placement=player0)
            return x0

        concrete_comp = trace(my_comp)
        constant_op = concrete_comp.operation("constant_0")
        assert constant_op == ops.ConstantOperation(
            placement_name="player0",
            name="constant_0",
            inputs={},
            value=values.TensorConstant(value=[1.0]),
            signature=ops.OpSignature({}, ty.TensorType(dtypes.float64)),
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
        assert constant_op == ops.LoadOperation(
            placement_name="player0",
            name="load_0",
            inputs={"key": "constant_0", "query": "constant_1"},
            signature=ops.OpSignature(
                {"key": ty.StringType(), "query": ty.StringType()},
                ty.TensorType(dtypes.float32),
            ),
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
        assert constant_op == ops.LoadOperation(
            placement_name="player0",
            name="load_0",
            inputs={"key": "constant_0", "query": "constant_1"},
            signature=ops.OpSignature(
                {"key": ty.StringType(), "query": ty.StringType()},
                ty.TensorType(dtypes.float32),
            ),
        )

    @parameterized.parameters(*_MOOSE_DTYPES)
    def test_tensor_arguments(self, tensor_dtype):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp(x: edsl.Argument(placement=player0, dtype=tensor_dtype)):
            y = edsl.constant(1.0, dtype=tensor_dtype, placement=player0)
            z = edsl.add(x, y, placement=player0)
            return z

        concrete_comp = trace(my_comp)

        assert concrete_comp == comp.Computation(
            operations={
                "x": ops.InputOperation(
                    placement_name="player0",
                    name="x",
                    inputs={},
                    signature=ops.OpSignature({}, ty.TensorType(tensor_dtype)),
                ),
                "constant_0": ops.ConstantOperation(
                    placement_name="player0",
                    name="constant_0",
                    inputs={},
                    value=values.TensorConstant(value=[1.0]),
                    signature=ops.OpSignature({}, ty.TensorType(tensor_dtype)),
                ),
                "add_0": ops.AddOperation(
                    placement_name="player0",
                    name="add_0",
                    inputs={"lhs": "x", "rhs": "constant_0"},
                    signature=ops.OpSignature(
                        {
                            "lhs": ty.TensorType(tensor_dtype),
                            "rhs": ty.TensorType(tensor_dtype),
                        },
                        ty.TensorType(tensor_dtype),
                    ),
                ),
                "output_0": ops.OutputOperation(
                    placement_name="player0",
                    name="output_0",
                    inputs={"value": "add_0"},
                    signature=ops.OpSignature(
                        {"value": ty.TensorType(tensor_dtype)},
                        ty.TensorType(tensor_dtype),
                    ),
                    tag="output_0",
                ),
            },
            placements={"player0": plc.HostPlacement(name="player0")},
        )

    @parameterized.parameters(
        (1, dtypes.int64, dtypes.int64),
        (1, dtypes.int32, dtypes.int32),
        (1.0, dtypes.float32, dtypes.float32),
        (1.0, dtypes.float64, dtypes.float64),
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
        if from_dtype is None:
            from_dtype = _npdtype_into_moose_dtype(input_value.dtype)
        assert cast_op == ops.CastOperation(
            placement_name="player0",
            name="cast_0",
            inputs={"x": "constant_0"},
            signature=ops.OpSignature(
                {"x": ty.TensorType(from_dtype)},
                ty.TensorType(into_dtype),
            ),
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

        from_dtype = _npdtype_into_moose_dtype(input_value.dtype)
        concrete_comp = trace(my_comp)
        cast_op = concrete_comp.operation("cast_0")
        assert cast_op == ops.CastOperation(
            placement_name="player0",
            name="cast_0",
            inputs={"x": "constant_0"},
            signature=ops.OpSignature(
                {"x": ty.TensorType(from_dtype)},
                ty.TensorType(dtype),
            ),
        )

    @parameterized.parameters(
        (build_ez_swap, {"alice": "re_l", "bob": "vincent", "carole": "pino"}),
        (build_mirr_swap, {"alice": "re_l", "bob": "vincent", "carole": "pino"}),
        (build_repl_swap, {"alice": "re_l", "bob": "vincent", "carole": "pino"}),
        (
            build_mirr_swap,
            {"alice": "re_l", "bob": "vincent", "carole": "pino", "mirrored": "proxy"},
        ),
        (
            build_repl_swap,
            {
                "alice": "re_l",
                "bob": "vincent",
                "carole": "pino",
                "replicated": "proxy",
            },
        ),
    )
    def test_role_swap(self, comp_builder, role_map):
        vanilla_comp = comp_builder()
        swapped_comp = vanilla_comp.with_role_map(role_map)
        vanilla = trace(vanilla_comp)
        swapped = trace(swapped_comp)
        for (vk, vv), (sk, sv) in zip(
            vanilla.placements.items(), swapped.placements.items()
        ):
            vkm = role_map.get(vk, None)
            vvm = role_map.get(vv.name, None)
            if vkm is not None or vvm is not None:
                assert vkm == sk, f"{vk} was mapped to  {sk}, not {vkm}"
                assert vvm == sv.name, f"{vv} was mapped to  {sv}, not {vvm}"

    def test_role_swap_cycle(self):
        vanilla_comp = build_ez_swap()
        role_swap = {"alice": "bob", "bob": "alice"}
        swapped_comp = vanilla_comp.with_role_map(role_swap)
        vanilla = trace(vanilla_comp)
        swapped = trace(swapped_comp)
        for op_name, op in vanilla.operations.items():
            if op.placement_name == "alice":
                assert swapped.operations[op_name].placement_name == "bob"
            elif op.placement_name == "bob":
                assert swapped.operations[op_name].placement_name == "alice"


def _npdtype_into_moose_dtype(npdtype):
    return edsl._NUMPY_DTYPES_MAP[npdtype]


if __name__ == "__main__":
    absltest.main()
