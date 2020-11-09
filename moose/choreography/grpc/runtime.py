import asyncio
import random
from typing import Dict
from typing import Optional
from typing import Union

from moose.cluster.cluster_spec import load_cluster_spec
from moose.compiler.computation import Computation
from moose.executor.executor import AsyncExecutor
from moose.executor.proxy import RemoteExecutor
from moose.logger import get_logger
from moose.networking.memory import ChannelManager
from moose.utils import load_certificate


class Runtime:
    def evaluate_computation(
        self, computation: Computation, placement_assignment: Dict
    ):
        sid = random.randrange(2 ** 32)
        tasks = [
            executor.run_computation(
                computation, placement=placement.name, session_id=sid
            )
            for placement, executor in placement_assignment.items()
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


class RemoteRuntime(Runtime):
    def __init__(
        self,
        cluster_spec: Union[Dict, str],
        ca_cert_filename=None,
        ident_cert_filename=None,
        ident_key_filename=None,
    ) -> None:
        if isinstance(cluster_spec, str):
            # assume `cluster_spec` is given as a path
            cluster_spec = load_cluster_spec(cluster_spec)
        ca_cert = load_certificate(ca_cert_filename)
        ident_cert = load_certificate(ident_cert_filename)
        ident_key = load_certificate(ident_key_filename)
        self.executors = {
            placement_name: RemoteExecutor(
                endpoint, ca_cert=ca_cert, ident_cert=ident_cert, ident_key=ident_key
            )
            for placement_name, endpoint in cluster_spec.items()
        }


class TestRuntime(Runtime):
    def __init__(self, workers) -> None:
        channel_manager = ChannelManager()
        self.executors = {
            placement_name: AsyncExecutor(
                name=placement_name, channel_manager=channel_manager
            )
            for placement_name in workers
        }


_RUNTIME: Optional[Runtime] = None


def set_runtime(runtime: Runtime):
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime():
    global _RUNTIME
    assert _RUNTIME is not None
    return _RUNTIME
