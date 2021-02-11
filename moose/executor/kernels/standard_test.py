import asyncio
import unittest

import numpy as np
from absl.testing import parameterized

from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import ShapeType
from moose.computation.standard import TensorType
from moose.edsl import base as edsl
from moose.edsl.tracer import trace
from moose.executor.executor import AsyncExecutor
from moose.testing import run_test_computation


class StandardKernelTest(parameterized.TestCase):
    def test_constant(self):
        player0 = edsl.host_placement("player0")
        player1 = edsl.host_placement("player1")

        @edsl.computation
        def my_comp():
            out = edsl.constant(5, placement=player0)
            res = edsl.save("result", out, placement=player1)
            return res

        comp_result = run_test_computation(trace(my_comp), [player0, player1])
        self.assertEqual(comp_result[player1]["result"], 5)

    def test_input(self):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.InputOperation(
                name="y",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.AddOperation(
                name="add",
                placement_name=alice.name,
                inputs={"lhs": "x", "rhs": "y"},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "add"},
            )
        )

        results = run_test_computation(comp, [alice], arguments={"x": 5, "y": 10})
        assert results[alice]["z"] == 15

    @parameterized.parameters(
        {"axis": axis, "expected_result": expected_result}
        for (axis, expected_result) in zip(
            [0, 1],
            [
                np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2),
                np.array([1, 2, 5, 6, 3, 4, 7, 8]).reshape(2, 4),
            ],
        )
    )
    def test_concatenate(self, axis, expected_result):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x_0",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.InputOperation(
                name="x_1",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.ConcatenateOperation(
                name="concatenate",
                placement_name=alice.name,
                axis=axis,
                inputs={"array0": "x_0", "array1": "x_1"},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "concatenate"},
            )
        )

        results = run_test_computation(
            comp,
            [alice],
            arguments={
                "x_0": np.array([[1, 2], [3, 4]]),
                "x_1": np.array([[5, 6], [7, 8]]),
            },
        )
        np.testing.assert_array_equal(results[alice]["z"], expected_result)

    @parameterized.parameters(
        {"op": op, "expected_result": expected_result}
        for (op, expected_result) in zip(
            [edsl.add, edsl.sub, edsl.mul, edsl.div], [7, 3, 10, 2.5]
        )
    )
    def test_op(self, op, expected_result):
        player0 = edsl.host_placement("player0")
        player1 = edsl.host_placement("player1")

        @edsl.computation
        def my_comp():
            out = op(
                edsl.constant(5, placement=player0),
                edsl.constant(2, placement=player0),
                placement=player0,
            )
            res = edsl.save("result", out, placement=player1)
            return res

        comp_result = run_test_computation(trace(my_comp), [player0, player1])
        self.assertEqual(comp_result[player1]["result"], expected_result)

    @parameterized.parameters(
        (np.array([[[[1], [2]]], [[[3], [4]]]]), None, (2, 2)),
        (np.array([[[[1], [2]]], [[[3], [4]]]]), 1, (2, 2, 1)),
        (np.array([[[[1], [2]]], [[[3], [4]]]]), (1, 3), (2, 2)),
    )
    def test_squeeze(self, input_array, axis, expected_shape):
        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                value=input_array,
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.SqueezeOperation(
                name="squeeze",
                placement_name=alice.name,
                inputs={"x": "x"},
                axis=axis,
                output_type=TensorType(dtype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "squeeze"},
            )
        )

        comp_result = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(comp_result[alice]["z"].shape, expected_shape)

    @parameterized.parameters(
        {"axis": ax, "expected_shape": exp}
        for ax, exp in [(1, (2, 1, 2)), (-1, (2, 2, 1)), ((1, 3), (2, 1, 2, 1))]
    )
    def test_unsqueeze(self, axis, expected_shape):
        x = np.array([[1, 2], [3, 4]])
        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                value=x,
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.ExpandDimsOperation(
                name="expand_dims",
                placement_name=alice.name,
                inputs={"x": "x"},
                axis=axis,
                output_type=TensorType(dtype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "expand_dims"},
            )
        )

        comp_result = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(comp_result[alice]["z"].shape, expected_shape)

    def test_inverse(self):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        x = np.array([[3, 2], [2, 3]], dtype=np.float32)
        expectation = np.linalg.inv(x)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                value=x,
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.InverseOperation(
                name="inverse",
                placement_name=alice.name,
                inputs={"x": "x"},
                output_type=TensorType(dtype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "inverse"},
            )
        )

        results = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(results[alice]["z"], expectation)

    @parameterized.parameters(
        {"dtype": dtype, "expected_result": expected_result}
        for (dtype, expected_result) in zip(
            [float, np.float64, int, np.int64],
            [
                np.ones((2, 2), float),
                np.ones((2, 2), np.float64),
                np.ones((2, 2), int),
                np.ones((2, 2), np.int64),
            ],
        )
    )
    def test_ones(self, dtype, expected_result):

        if isinstance(dtype, (int, np.int64)):
            datatype = "int64"
        else:
            datatype = "float"

        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="shape_op",
                placement_name=alice.name,
                inputs={},
                value=[2, 2],
                output_type=ShapeType(),
            )
        )
        comp.add_operation(
            standard_dialect.OnesOperation(
                name="x",
                placement_name=alice.name,
                dtype=dtype,
                inputs={"shape": "shape_op"},
                output_type=TensorType(dtype=datatype),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="y",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "x"},
            )
        )

        results = run_test_computation(comp, [alice])
        assert results[alice]["y"].dtype == expected_result.dtype
        np.testing.assert_array_equal(results[alice]["y"], expected_result)

    @parameterized.parameters(
        (standard_dialect.SumOperation, "sum", None, 10),
        (standard_dialect.SumOperation, "sum", 0, np.array([4, 6])),
        (standard_dialect.SumOperation, "sum", (0, 1), 10),
        (standard_dialect.MeanOperation, "mean", None, 2.5),
        (standard_dialect.MeanOperation, "mean", 0, np.array([2, 3])),
        (standard_dialect.MeanOperation, "mean", (0, 1), 2.5),
    )
    def test_reduce_op(self, reduce_op_cls, reduce_op_name, axis, expected_result):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            reduce_op_cls(
                name=reduce_op_name,
                placement_name=alice.name,
                axis=axis,
                inputs={"x": "x"},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": reduce_op_name},
            )
        )

        results = run_test_computation(
            comp, [alice], arguments={"x": np.array([[1, 2], [3, 4]])}
        )
        np.testing.assert_array_equal(results[alice]["z"], expected_result)

    @parameterized.parameters(
        {"axes": axes, "expected_result": expected_result}
        for (axes, expected_result) in zip(
            [None, (1, 0), (0, 1)],
            [
                np.array([[1, 3], [2, 4]]),
                np.array([[1, 3], [2, 4]]),
                np.array([[1, 2], [3, 4]]),
            ],
        )
    )
    def test_transpose(self, axes, expected_result):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.TransposeOperation(
                name="transpose",
                placement_name=alice.name,
                axes=axes,
                inputs={"x": "x"},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "transpose"},
            )
        )

        results = run_test_computation(
            comp, [alice], arguments={"x": np.array([[1, 2], [3, 4]])}
        )
        np.testing.assert_array_equal(results[alice]["z"], expected_result)

    @parameterized.parameters(
        (np.array(1), False, np.array([[1]])),
        (np.ones(shape=(3,)), False, np.ones(shape=(1, 3))),
        (np.ones(shape=(3,)), True, np.ones(shape=(3, 1))),
        (np.ones(shape=(1, 1)), False, np.ones(shape=(1, 1))),
    )
    def test_atleast_2d(self, input, to_column_vector, expected_result):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.Atleast2DOperation(
                name="atleast2d",
                placement_name=alice.name,
                inputs={"x": "x"},
                to_column_vector=to_column_vector,
                output_type=TensorType(dtype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "atleast2d"},
            )
        )

        results = run_test_computation(comp, [alice], arguments={"x": input})
        np.testing.assert_array_equal(results[alice]["z"], expected_result)

    def test_exception(self):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype="int64"),
            )
        )

        # there is no networking so run_computation below will raise
        # exception
        comp.add_operation(
            standard_dialect.SendOperation(
                name="send_x",
                placement_name=alice.name,
                inputs={"value": "x"},
                sender=alice.name,
                receiver=alice.name,
                rendezvous_key="0123456789",
            )
        )

        executor = AsyncExecutor(networking=None, storage=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
            arguments={"x": 5},
        )
        with self.assertRaises(Exception):
            asyncio.get_event_loop().run_until_complete(task)


if __name__ == "__main__":
    unittest.main()
