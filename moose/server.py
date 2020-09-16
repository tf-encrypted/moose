import asyncio
from collections import defaultdict

from grpc.experimental import aio

from moose.compiler.computation import Computation
from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc


class ExecutorServicer(executor_pb2_grpc.ExecutorServicer):
    def __init__(self, executor=None):
        self.buffer = defaultdict(asyncio.get_event_loop().create_future)
        self.executor = executor

    async def GetValue(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        value = await self.buffer[key]
        return executor_pb2.GetValueResponse(value=value)

    async def SetValue(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        self.buffer[key].set_result(request.value)
        return executor_pb2.SetValueResponse()

    async def RunComputation(self, request, context):
        computation = Computation.deserialize(request.computation)
        placement = request.placement
        session_id = request.session_id
        await self.executor.run_computation(computation, placement, session_id)
        return executor_pb2.RunComputationResponse()


class Server:
    def __init__(self, host, port, executor):
        self._endpoint = f"{host}:{port}"
        self._servicer = ExecutorServicer(executor)
        self._server = None

    async def start(self):
        self._server = aio.server()
        executor_pb2_grpc.add_ExecutorServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(self._endpoint)
        await self._server.start()

    async def wait(self):
        await self._server.wait_for_termination()
        self._server = None
