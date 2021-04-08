import asyncio

from absl.testing import parameterized

from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.dtypes import int64
from moose.computation.host import HostPlacement
from moose.computation.standard import IntConstant
from moose.computation.standard import TensorType
from moose.executor.executor import AsyncExecutor


class ExecutorTest(parameterized.TestCase):
    def test_rerun(self):
        executor = AsyncExecutor(networking=None, storage=None)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                value=IntConstant(value=12345),
                output_type=TensorType(dtype=int64),
            )
        )

        executor = AsyncExecutor(networking=None, storage=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="01234",
        )
        asyncio.get_event_loop().run_until_complete(task)

        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="01234",
        )
        with self.assertRaises(Exception):
            asyncio.get_event_loop().run_until_complete(task)

        executor = AsyncExecutor(networking=None, storage=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="56789",
        )
        asyncio.get_event_loop().run_until_complete(task)

        task = executor.run_computation(
            comp,
            placement_instantiation={alice.name: alice.name},
            placement=alice.name,
            session_id="56789",
        )
        with self.assertRaises(Exception):
            asyncio.get_event_loop().run_until_complete(task)
