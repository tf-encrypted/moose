import asyncio
import random
from typing import Dict

from moose.computation.base import Computation
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.logger import get_tracer
from moose.networking.memory import Networking
from moose.storage.memory import MemoryDataStore

from moose import edsl
from moose.computation.utils import serialize_computation

from pymoose import MooseLocalRuntime

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

# TODO [Yann] Rename if we decide to keep
# We might want to subclass MooseLocalRuntime instead?
class NewTestRuntime:
    def __init__(self, executors_storage: dict):
        self._executors_storage = executors_storage
        self._runtime = MooseLocalRuntime(self._executors_storage)

    def evaluate_computation(self, computation, arguments={}, ring=128):
        concrete_comp, outputs_name = edsl.trace_and_compile(computation, ring=ring)
        comp_bin = serialize_computation(concrete_comp)
        comp_outputs = self._runtime.evaluate_computation(comp_bin, arguments)
        outputs = [comp_outputs.get(output_name) for output_name in outputs_name]
        return outputs

    def get_value_from_storage(self, placement, key):
        return self._runtime.get_value_from_storage(placement, key)
        
