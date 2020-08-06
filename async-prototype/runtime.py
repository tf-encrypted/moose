import asyncio
import random
from typing import Dict
from typing import Optional

from computation import Computation
from executor import RemoteExecutor


class Runtime:
    def __init__(self, role_assignment: Dict) -> None:
        self.role_assignment = role_assignment

    def evaluate_computation(self, comp: Computation):
        loop = asyncio.get_event_loop()
        sid = random.randrange(2 ** 32)
        tasks = [
            executor.run_computation(
                comp, role=role.name, session_id=sid, event_loop=loop
            )
            for role, executor in self.role_assignment.items()
        ]
        joint_task = asyncio.wait(tasks)
        loop.run_until_complete(joint_task)


class RemoteRuntime:
    def __init__(self, cluster_spec: Dict) -> None:
        self.remote_executers = self.create_remote_executers(cluster_spec)

    def evaluate_computation(self, comp: Computation):
        loop = asyncio.get_event_loop()
        sid = random.randrange(2 ** 32)
        tasks = [
            remote_executor.run_computation(comp, role=role, session_id=sid)
            for role, remote_executor in self.remote_executers.items()
        ]
        joint_task = asyncio.wait(tasks)
        loop.run_until_complete(joint_task)

    def create_remote_executers(self, clusper_spec):
        remote_executers = {
            role: RemoteExecutor(endpoint) for role, endpoint in clusper_spec.items()
        }
        return remote_executers


_RUNTIME: Optional[Runtime] = None


def set_runtime(runtime: Runtime):
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime():
    global _RUNTIME
    assert _RUNTIME is not None
    return _RUNTIME
