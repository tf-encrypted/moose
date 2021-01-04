import asyncio
import logging
import unittest

import numpy as np
from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import TensorType
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.memory import Networking

get_logger().setLevel(level=logging.DEBUG)


class ReplicatedProtocolsTest(parameterized.TestCase):
    @parameterized.parameters(
        (lambda x, y: x + y, standard_dialect.AddOperation),
        (lambda x, y: x - y, standard_dialect.SubOperation),
        # the following will work only after we can do fix point multiplication
        # without special encoding
        # (lambda x, y: x * y, standard_dialect.MulOperation), 
    )
    def test_bin_op(self, numpy_lmbd, replicated_std_op):
        comp = Computation(operations={}, placements={})

        alice = HostPlacement(name="alice")
        bob = HostPlacement(name="bob")
        carole = HostPlacement(name="carole")
        rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])

        comp.add_placement(alice)
        comp.add_placement(bob)
        comp.add_placement(carole)
        comp.add_placement(rep)

        x = np.array([10], dtype=np.float64)
        y = np.array([20], dtype=np.float64)

        z = numpy_lmbd(x, y)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=x,
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="bob_input",
                value=y,
                placement_name=bob.name,
                inputs={},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            replicated_std_op(
                name="rep_op",
                placement_name=rep.name,
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"value": "rep_op"},
                placement_name=carole.name,
                key="result",
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output", placement_name=carole.name, inputs={"value": "save"},
            )
        )

        compiler = Compiler()

        comp = compiler.run_passes(comp)

        placement_instantiation = {
            alice: alice.name,
            bob: bob.name,
            carole: carole.name,
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

        np.testing.assert_array_equal(z, placement_executors[carole].store["result"])


if __name__ == "__main__":
    unittest.main()
