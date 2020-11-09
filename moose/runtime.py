import asyncio
import random
from typing import Dict

from moose.channels.memory import ChannelManager
from moose.compiler.computation import Computation
from moose.executor.executor import AsyncExecutor
from moose.executor.proxy import RemoteExecutor
from moose.logger import get_logger
from moose.utils import load_certificate

# TODO(Morten) bring the below back as an interface?
# class Runtime:
#     def evaluate_computation(
#         self, computation: Computation, placement_assignment: Dict
#     ):


class RemoteRuntime:
    def __init__(
        self, ca_cert_filename=None, ident_cert_filename=None, ident_key_filename=None,
    ) -> None:
        self.ca_cert = load_certificate(ca_cert_filename)
        self.ident_cert = load_certificate(ident_cert_filename)
        self.ident_key = load_certificate(ident_key_filename)
        self.existing_executors = dict()

    def evaluate_computation(
        self, computation: Computation, placement_instantiation: Dict
    ):
        placement_executors = dict()
        for placement, endpoint in placement_instantiation.items():
            if endpoint not in self.existing_executors:
                self.existing_executors[endpoint] = RemoteExecutor(
                    endpoint,
                    ca_cert=self.ca_cert,
                    ident_cert=self.ident_cert,
                    ident_key=self.ident_key,
                )
            placement_executors[placement] = self.existing_executors[endpoint]

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


class TestRuntime:
    def __init__(self) -> None:
        self.channel_manager = ChannelManager()
        self.existing_executors = dict()

    def evaluate_computation(
        self, computation: Computation, placement_instantiation: Dict
    ):
        placement_executors = dict()
        for placement, name in placement_instantiation.items():
            if name not in self.existing_executors:
                self.existing_executors[name] = AsyncExecutor(
                    name=name, channel_manager=self.channel_manager
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
