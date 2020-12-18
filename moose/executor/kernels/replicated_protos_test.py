import asyncio
import logging
import unittest

from absl.testing import parameterized

from moose.computation import standard as standard_dialect
from moose.computation import replicated as replicated_ops
from moose.computation.replicated import ReplicatedSetupType
from moose.computation.replicated import ReplicatedTensorType
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import TensorType
from moose.edsl import replicated_placement
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
from moose.logger import get_logger
from moose.runtime import TestRuntime as Runtime

get_logger().setLevel(level=logging.DEBUG)


class ReplicatedProtocolsTest(parameterized.TestCase):
    def test_add(self):
        comp = Computation(operations={}, placements={})

        alice = HostPlacement(name="alice")
        bob = HostPlacement(name="bob")
        carole = HostPlacement(name="carole")
        rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])
        comp.add_placement(alice)
        comp.add_placement(bob)
        comp.add_placement(carole)
        comp.add_placement(rep)

        comp.add_operation(
            standard_dialect.InputOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="int64"),
            )
        )

        comp.add_operation(
            replicated_ops.SetupOperation(
                name="replicated_setup_0",
                inputs={},
                placement_name="rep",
                output_type=ReplicatedSetupType(),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x_shape",
                placement_name=alice.name,
                inputs={},
                value=(2, 2),
                output_type=standard_dialect.ShapeType(),
            )
        )

        executor = AsyncExecutor(networking=None)

        placement_instantiation= {
            alice:alice.name,
            bob:bob.name,
            carole:carole.name
        }

        placement_executors = {
            alice: AsyncExecutor(networking=None),
            bob: AsyncExecutor(networking=None),
            carole: AsyncExecutor(networking=None),
        }

        tasks = [
            executor.run_computation(
                comp,
                placement_instantiation=placement_instantiation,
                placement=placement,
                session_id="0123456789",
            )
            for placement, executor in placement_executors.items()
        ]
        joint_task = asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        done, _ = asyncio.get_event_loop().run_until_complete(joint_task)
 
        # task = executor.run_computation(
        #     comp,
        #    placement=alice.name,
        #     session_id="0123456789",
        #     arguments={"x": 5, "y": 10},
        # asyncio.get_event_loop().run_until_complete(task)
        print(placement_executors[alice].store)
        assert placement_executors[alice].store["x_shape"] == (2,2)


if __name__ == "__main__":
    unittest.main()
