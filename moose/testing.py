import asyncio
import random
from typing import Dict

from pymoose import LocalRuntime

from moose import edsl
from moose.computation.base import Computation
from moose.computation.utils import serialize_computation
from moose.deprecated.executor.executor import AsyncExecutor
from moose.deprecated.networking.memory import Networking
from moose.deprecated.storage.memory import MemoryDataStore
from moose.logger import get_logger
from moose.logger import get_tracer


class TestRuntime:
    def __init__(self, networking=None, backing_executors=None) -> None:
        self.networking = networking or Networking()
        self.existing_executors = backing_executors or dict()

    def evaluate_computation(
        self,
        computation: Computation,
        placement_instantiation: Dict,
        arguments: Dict = {},
    ):
        placement_instantiation = {
            placement.name if not isinstance(placement, str) else placement: endpoint
            for placement, endpoint in placement_instantiation.items()
        }
        placement_executors = dict()
        for placement, name in placement_instantiation.items():
            if name not in self.existing_executors:
                self.existing_executors[name] = AsyncExecutor(
                    networking=self.networking, storage=MemoryDataStore()
                )
            placement_executors[placement] = self.existing_executors[name]

        sid = random.randrange(2 ** 32)

        with get_tracer().start_as_current_span("eval") as span:
            span.set_attribute("moose.session_id", sid)
            tasks = [
                executor.run_computation(
                    computation,
                    placement_instantiation=placement_instantiation,
                    placement=placement,
                    session_id=sid,
                    arguments=arguments,
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


def run_test_computation(computation, players, arguments={}):
    runtime = TestRuntime()
    runtime.evaluate_computation(
        computation,
        placement_instantiation={player: player.name for player in players},
        arguments=arguments,
    )
    return {
        player: runtime.get_executor(player.name).storage.store for player in players
    }


class LocalMooseRuntime(LocalRuntime):
    def __new__(cls, *, identities=None, storage_mapping=None):
        if identities is None and storage_mapping is None:
            raise ValueError(
                "Must provide either a list of identities or a mapping of identities "
                "to executor storage dicts."
            )
        elif storage_mapping is not None and identities is not None:
            assert storage_mapping.keys() == identities
        elif identities is not None:
            storage_mapping = {identity: {} for identity in identities}
        return LocalRuntime.__new__(LocalMooseRuntime, storage_mapping=storage_mapping)

    def evaluate_computation(
        self, computation, role_assignment, arguments=None, ring=128
    ):
        if arguments is None:
            arguments = {}
        concrete_comp = edsl.trace_and_compile(computation, ring=ring)
        comp_bin = serialize_computation(concrete_comp)
        comp_outputs = super().evaluate_computation(
            comp_bin, role_assignment, arguments
        )
        outputs = list(dict(sorted(comp_outputs.items())).values())
        return outputs

    def evaluate_compiled(self, comp_bin, role_assignment, arguments=None, ring=128):
        if arguments is None:
            arguments = {}
        comp_outputs = super().evaluate_compiled(comp_bin, role_assignment, arguments)
        outputs = list(dict(sorted(comp_outputs.items())).values())
        return outputs

    def read_value_from_storage(self, identity, key):
        return super().read_value_from_storage(identity, key)

    def write_value_to_storage(self, identity, key, value):
        return super().write_value_to_storage(identity, key, value)
