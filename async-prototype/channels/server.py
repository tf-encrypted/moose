from collections import defaultdict
from concurrent import futures
import time
import math
import logging

import asyncio
from grpc.experimental import aio
import threading

from protos import secure_channel_pb2
from protos import secure_channel_pb2_grpc


class SecureChannelServicer(secure_channel_pb2_grpc.SecureChannelServicer):
    def __init__(self):
        self.buffer = defaultdict(asyncio.get_event_loop().create_future)

    async def GetValue(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        value = await self.buffer[key]
        return secure_channel_pb2.Value(value=value)

    async def AddValueToBuffer(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        self.buffer[key].set_result(request.value)
        return secure_channel_pb2.Empty()


class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._endpoint = self.host + ":" + self.port
        self._servicer = SecureChannelServicer()
        self.server = None

    async def start(self):
        self.server = aio.server()
        secure_channel_pb2_grpc.add_SecureChannelServicer_to_server(
            self._servicer, self.server
        )
        self.server.add_insecure_port(self._endpoint)
        await self.server.start()

    async def wait(self):
        await self.server.wait_for_termination()
        self.server = None
