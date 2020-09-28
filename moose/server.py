from grpc.experimental import aio

from moose.channels.grpc import ChannelManager
from moose.compiler.computation import Computation
from moose.executor.executor import KernelBasedExecutor
from moose.logger import get_logger
from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc
from moose.storage import AsyncStore


class ExecutorServicer(executor_pb2_grpc.ExecutorServicer):
    def __init__(self, executor, buffer):
        self.buffer = buffer
        self.executor = executor

    async def GetValue(self, request, context):
        get_logger().debug(
            f"Received value for key {request.rendezvous_key} "
            f"for session {request.session_id}"
        )
        key = (request.session_id, request.rendezvous_key)
        value = await self.buffer.get(key)
        return executor_pb2.GetValueResponse(value=value)

    async def SetValue(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        await self.buffer.put(key, request.value)
        return executor_pb2.SetValueResponse()

    async def RunComputation(self, request, context):
        computation = Computation.deserialize(request.computation)
        placement = request.placement
        session_id = request.session_id
        await self.executor.run_computation(computation, placement, session_id)
        return executor_pb2.RunComputationResponse()


class Server:
    def __init__(self, host, port, cluster_spec):
        channel_manager = ChannelManager(cluster_spec)
        executor = KernelBasedExecutor(name="remote", channel_manager=channel_manager)
        self._endpoint = f"{host}:{port}"
        self._servicer = ExecutorServicer(executor, AsyncStore())
        self._server = None

    async def start(self):
        self._server = aio.server()
        executor_pb2_grpc.add_ExecutorServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(self._endpoint)
        await self._server.start()

    async def wait(self):
        await self._server.wait_for_termination()
        self._server = None
