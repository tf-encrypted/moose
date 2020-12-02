import asyncio
import random
from typing import Dict

from moose.computation.base import Computation
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.memory import Networking


class TestRuntime:
    def __init__(self) -> None:
        self.networking = Networking()
        self.existing_executors = dict()

    def evaluate_computation(
        self, computation: Computation, placement_instantiation: Dict
    ):
        placement_instantiation = {
            placement.name if not isinstance(placement, str) else placement: endpoint
            for placement, endpoint in placement_instantiation.items()
        }
        placement_executors = dict()
        for placement, name in placement_instantiation.items():
            if name not in self.existing_executors:
                self.existing_executors[name] = AsyncExecutor(
                    networking=self.networking
                )
            placement_executors[placement] = self.existing_executors[name]

        sid = random.randrange(2 ** 32)
        tasks = [
            executor.run_computation(
                computation,
                placement_instantiation=placement_instantiation,
                placement=placement,
                session_id=sid,
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

    def get_executor(self, executor_name):
        return self.existing_executors[executor_name]
