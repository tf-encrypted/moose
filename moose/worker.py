from grpc.experimental import aio

from moose.channels.grpc import ChannelManager
from moose.compiler.computation import Computation
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.protos import channel_manager_pb2
from moose.protos import channel_manager_pb2_grpc
from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc


class ExecutorServicer(executor_pb2_grpc.ExecutorServicer):
    def __init__(self, executor):
        self.executor = executor

    async def RunComputation(self, request, context):
        computation = Computation.deserialize(request.computation)
        placement = request.placement
        session_id = request.session_id
        await self.executor.run_computation(computation, placement, session_id)
        return executor_pb2.RunComputationResponse()


class ChannelManagerServicer(channel_manager_pb2_grpc.ChannelManagerServicer):
    def __init__(self, channel_manager):
        self.channel_manager = channel_manager

    async def GetValue(self, request, context):
        get_logger().debug(
            f"Received value for key {request.rendezvous_key} "
            f"for session {request.session_id}"
        )
        key = (request.session_id, request.rendezvous_key)
        value = await self.channel_manager.buffer.get(key)  # TODO(Morten) leaking impl
        return channel_manager_pb2.GetValueResponse(value=value)


class Worker:
    def __init__(self, name, host, port, cluster_spec):
        # core components
        channel_manager = ChannelManager(cluster_spec)
        executor = AsyncExecutor(name=name, channel_manager=channel_manager)

        # set up gRPC server exposing core components
        self._server = aio.server()
        self._server.add_insecure_port(f"{host}:{port}")
        executor_pb2_grpc.add_ExecutorServicer_to_server(
            ExecutorServicer(executor), self._server,
        )
        channel_manager_pb2_grpc.add_ChannelManagerServicer_to_server(
            ChannelManagerServicer(channel_manager), self._server,
        )

    async def start(self):
        await self._server.start()

    async def wait(self):
        await self._server.wait_for_termination()
