import asyncio
import logging
import pathlib
import unittest

import numpy as np
from absl.testing import parameterized

from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import TensorType
from moose.edsl.base import add
from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import div
from moose.edsl.base import function
from moose.edsl.base import host_placement
from moose.edsl.base import mul
from moose.edsl.base import run_program
from moose.edsl.base import save
from moose.edsl.base import sub
from moose.edsl.tracer import trace
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.runtime import TestRuntime as Runtime

get_logger().setLevel(level=logging.DEBUG)


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


class ExecutorTest(parameterized.TestCase):
    def test_call_python_function(self):
        player0, player1 = _create_test_players(2)

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            out = add_one(constant(3, placement=player0), placement=player0)
            res = save(out, "result", placement=player1)
            return res

        comp_result = _run_computation(my_comp, [player0, player1])
        self.assertEqual(comp_result["result"], 4)

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

    def test_run_program(self):
        player0, player1, player2 = _create_test_players(3)
        test_fixtures_file = str(
            pathlib.Path(__file__)
            .parent.absolute()
            .joinpath("executor_test_fixtures.py")
        )

        @computation
        def my_comp():
            c0 = constant(3, placement=player0)
            c1 = constant(2, placement=player0)
            out = run_program("python", [test_fixtures_file], c0, c1, placement=player1)
            res = save(out, "result", placement=player2)
            return res

        comp_result = _run_computation(my_comp, [player0, player1, player2])
        self.assertEqual(comp_result["result"], 6)

    @parameterized.parameters(
        (
            lambda plc: ring_dialect.RingAddOperation(
                name="ring_add",
                placement_name=plc.name,
                inputs={"lhs": "x", "rhs": "y"},
            ),
            "ring_add",
            lambda x, y: x + y,
        ),
        (
            lambda plc: ring_dialect.RingMulOperation(
                name="ring_mul",
                placement_name=plc.name,
                inputs={"lhs": "x", "rhs": "y"},
            ),
            "ring_mul",
            lambda x, y: x * y,
        ),
        (
            lambda plc: ring_dialect.RingSubOperation(
                name="ring_sub",
                placement_name=plc.name,
                inputs={"lhs": "x", "rhs": "y"},
            ),
            "ring_sub",
            lambda x, y: x - y,
        ),
    )
    def test_ring_binop_invocation(self, ring_op_lmbd, ring_op_name, numpy_lmbd):
        a = np.array([3], dtype=np.uint64)
        b = np.array([2], dtype=np.uint64)
        c = numpy_lmbd(a, b)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                value=a,
                output_type=standard_dialect.TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="y",
                placement_name=alice.name,
                inputs={},
                value=b,
                output_type=standard_dialect.TensorType(datatype="float"),
            )
        )
        comp.add_operation(ring_op_lmbd(alice))
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": ring_op_name},
                key="z",
            )
        )

        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice: alice.name},
            placement=alice.name,
            session_id="0123456789",
        )
        asyncio.get_event_loop().run_until_complete(task)
        np.testing.assert_array_equal(c, executor.store["z"])

    @parameterized.parameters(
        ([[1, 2], [3, 4]], [[1, 0], [0, 1]]),
        ([[1, 2], [3, 4]], [1, 1]),
        ([1, 1], [[1, 2], [3, 4]]),
    )
    def test_ring_dot_invocation(self, a, b):
        x = np.array(a, dtype=np.uint64)
        y = np.array(b, dtype=np.uint64)
        exp = np.dot(x, y)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                value=x,
                output_type=standard_dialect.TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="y",
                placement_name=alice.name,
                inputs={},
                value=y,
                output_type=standard_dialect.TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            ring_dialect.RingDotOperation(
                name="ring_dot",
                placement_name=alice.name,
                inputs={"lhs": "x", "rhs": "y"},
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "ring_dot"},
                key="z",
            )
        )

        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice: alice.name},
            placement=alice.name,
            session_id="0123456789",
        )
        asyncio.get_event_loop().run_until_complete(task)
        np.testing.assert_array_equal(exp, executor.store["z"])


if __name__ == "__main__":
    unittest.main()
