from collections import defaultdict
from concurrent import futures
import time
import math
import logging

import asyncio
from grpc.experimental import aio
import threading

from protos import executor_pb2
from protos import executor_pb2_grpc

from computation import Computation


class ExecutorServicer(executor_pb2_grpc.ExecutorServicer):
    def __init__(self, executor=None):
        self.buffer = defaultdict(asyncio.get_event_loop().create_future)
        self.executor = executor
        self._loop = asyncio.get_event_loop()

    async def GetValue(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        value = await self.buffer[key]
        return executor_pb2.ValueResponse(value=value)

    async def AddValueToBuffer(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        self.buffer[key].set_result(request.value)
        return executor_pb2.Empty()

    async def Compute(self, request, context):
        computation = Computation.deserialize(request.computation)
        role = request.role
        session_id = request.session_id
        await self.executor.run_computation(computation, role, session_id, self._loop)
        return executor_pb2.ComputeResponse()


class Server:
    def __init__(self, host, port, executor=None):
        self.host = host
        self.port = port
        self._endpoint = self.host + ":" + self.port
        self._servicer = ExecutorServicer(executor)
        self.server = None

    async def start(self):
        self.server = aio.server()
        executor_pb2_grpc.add_ExecutorServicer_to_server(self._servicer, self.server)
        self.server.add_insecure_port(self._endpoint)
        await self.server.start()

    async def wait(self):
        await self.server.wait_for_termination()
        self.server = None
