import asyncio
import unittest

import numpy as np
from absl.testing import parameterized

from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.edsl.base import host_placement
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


class RingKernelTest(parameterized.TestCase):
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

    def test_fill(self):
        expected = np.full((2, 2), 1)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x_shape",
                placement_name=alice.name,
                inputs={},
                value=(2, 2),
                output_type=standard_dialect.ShapeType(),
            )
        )
        comp.add_operation(
            ring_dialect.FillTensorOperation(
                name="x",
                placement_name=alice.name,
                value=1,
                inputs={"shape": "x_shape"},
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "x"},
                key="x_filled",
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
        np.testing.assert_array_equal(expected, executor.store["x_filled"])


if __name__ == "__main__":
    unittest.main()
