import asyncio
import logging
import unittest
import numpy as np

from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.compiler.replicated import ReplicatedFromStandardOpsPass
from moose.compiler.replicated import ReplicatedShareRevealPass


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

from moose.networking.memory import Networking

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
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=np.array([10], dtype=np.float64),
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="bob_input",
                value=np.array([20], dtype=np.float64),
                placement_name=bob.name,
                inputs={},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.AddOperation(
                name="add",
                placement_name=rep.name,
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output_add", inputs={"value": "add"}, placement_name=carole.name
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=carole.name,
                inputs={"value": "output_add"},
                key="final_output",
            )
        )

        compiler = Compiler()

        comp = compiler.run_passes(comp, render=True)

        placement_instantiation= {
            alice:alice.name,
            bob:bob.name,
            carole:carole.name
        }

        networking = Networking()
        placement_executors = {
            alice: AsyncExecutor(networking=networking),
            bob: AsyncExecutor(networking=networking),
            carole: AsyncExecutor(networking=networking),
        }

        tasks = [
            executor.run_computation(
                comp,
                placement_instantiation=placement_instantiation,
                placement=placement.name,
                session_id="0123456789",
            )
            for placement, executor in placement_executors.items()
        ]
        joint_task = asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        done, _ = asyncio.get_event_loop().run_until_complete(joint_task)
        exceptions = [task.exception() for task in done if task.exception()]
        for e in exceptions:
            get_logger().exception(e)
        if exceptions:
            raise Exception(
                "One or more errors evaluting the computation, see log for details"
            )
        
        print("carole ***", placement_executors[carole].store["final_output"])

        assert 0 == 1


if __name__ == "__main__":
    unittest.main()
