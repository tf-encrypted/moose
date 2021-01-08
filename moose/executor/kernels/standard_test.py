import asyncio
import unittest

import numpy as np
from absl.testing import parameterized

from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import TensorType
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import div
from moose.edsl.base import host_placement
from moose.edsl.base import mul
from moose.edsl.base import save
from moose.edsl.base import sub
from moose.edsl.tracer import trace
from moose.executor.executor import AsyncExecutor
from moose.runtime import TestRuntime as Runtime


def _create_test_players(number_of_players=2):
    return [host_placement(name=f"player_{i}") for i in range(number_of_players)]


def _run_computation(comp, players):
    runtime = Runtime()
    placement_instantiation = {player: player.name for player in players}
    concrete_comp = trace(comp)
    runtime.evaluate_computation(
        concrete_comp, placement_instantiation=placement_instantiation
    )
    return runtime.get_executor(players[-1].name).store


class StandardKernelTest(parameterized.TestCase):
    def test_constant(self):
        player0, player1 = _create_test_players(2)

        @computation
        def my_comp():
            out = constant(5, placement=player0)
            res = save(out, "result", placement=player1)
            return res

        comp_result = _run_computation(my_comp, [player0, player1])
        self.assertEqual(comp_result["result"], 5)

    def test_input(self):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.InputOperation(
                name="y",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.AddOperation(
                name="add",
                placement_name=alice.name,
                inputs={"lhs": "x", "rhs": "y"},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save", placement_name=alice.name, inputs={"value": "add"}, key="z"
            )
        )

        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
            arguments={"x": 5, "y": 10},
        )
        asyncio.get_event_loop().run_until_complete(task)
        assert executor.store["z"] == 15

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
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.InputOperation(
                name="x_1",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.ConcatenateOperation(
                name="concatenate",
                placement_name=alice.name,
                axis=axis,
                inputs={"array0": "x_0", "array1": "x_1"},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "concatenate"},
                key="z",
            )
        )
        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
            arguments={
                "x_0": np.array([[1, 2], [3, 4]]),
                "x_1": np.array([[5, 6], [7, 8]]),
            },
        )
        asyncio.get_event_loop().run_until_complete(task)
        np.testing.assert_array_equal(executor.store["z"], expected_result)

    @parameterized.parameters(
        {"op": op, "expected_result": expected_result}
        for (op, expected_result) in zip([add, sub, mul, div], [7, 3, 10, 2.5])
    )
    def test_op(self, op, expected_result):
        player0, player1 = _create_test_players(2)

        @computation
        def my_comp():
            out = op(
                constant(5, placement=player0),
                constant(2, placement=player0),
                placement=player0,
            )
            res = save(out, "result", placement=player1)
            return res

        comp_result = _run_computation(my_comp, [player0, player1])
        self.assertEqual(comp_result["result"], expected_result)

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
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.InverseOperation(
                name="inverse",
                placement_name=alice.name,
                inputs={"x": "x"},
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "inverse"},
                key="z",
            )
        )

        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
        )
        asyncio.get_event_loop().run_until_complete(task)
        np.testing.assert_array_equal(executor.store["z"], expectation)

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
            standard_dialect.OnesOperation(
                name="x",
                placement_name=alice.name,
                shape=(2, 2),
                dtype=dtype,
                inputs={},
                output_type=TensorType(datatype=datatype),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save", placement_name=alice.name, inputs={"value": "x"}, key="y",
            )
        )
        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
        )
        asyncio.get_event_loop().run_until_complete(task)
        assert executor.store["y"].dtype == expected_result.dtype
        np.testing.assert_array_equal(executor.store["y"], expected_result)

    def test_square(self):
        expected_result = np.array([1, 4, 9, 16])

        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.SquareOperation(
                name="square",
                placement_name=alice.name,
                inputs={"x": "x"},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "square"},
                key="z",
            )
        )
        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
            arguments={"x": np.array([1, 2, 3, 4])},
        )
        asyncio.get_event_loop().run_until_complete(task)
        np.testing.assert_array_equal(executor.store["z"], expected_result)

    @parameterized.parameters(
        {"axis": axis, "expected_result": expected_result}
        for (axis, expected_result) in zip(
            [None, 0, (0, 1)], [10, np.array([4, 6]), 10],
        )
    )
    def test_sum(self, axis, expected_result):
        comp = Computation(operations={}, placements={})

        alice = comp.add_placement(HostPlacement(name="alice"))

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.SumOperation(
                name="sum",
                placement_name=alice.name,
                axis=axis,
                inputs={"x": "x"},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "sum"},
                key="z",
            )
        )
        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
            arguments={"x": np.array([[1, 2], [3, 4]])},
        )
        asyncio.get_event_loop().run_until_complete(task)
        np.testing.assert_array_equal(executor.store["z"], expected_result)

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
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.TransposeOperation(
                name="transpose",
                placement_name=alice.name,
                axes=axes,
                inputs={"x": "x"},
                output_type=TensorType(datatype="int64"),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "transpose"},
                key="z",
            )
        )
        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="0123456789",
            arguments={"x": np.array([[1, 2], [3, 4]])},
        )
        asyncio.get_event_loop().run_until_complete(task)
        np.testing.assert_array_equal(executor.store["z"], expected_result)


if __name__ == "__main__":
    unittest.main()
