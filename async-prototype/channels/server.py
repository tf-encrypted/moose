from collections import defaultdict
from concurrent import futures
import time
import math
import logging

import asyncio
# import grpc
from grpc.experimental import aio 

from channels import secure_channel_pb2
from channels import secure_channel_pb2_grpc


# async def get_value(buffer, key):
#     key = (key.session_id, key.rendezvous_key)
#     value = await buffer.get(key)
#     # if value == None:
#     #     buffer[key] = asyncio.Future()
#     #     return What?
#     # else:
#     return secure_channel_pb2.Value(value=value)



class SecureChannelServicer(secure_channel_pb2_grpc.SecureChannelServicer):
    def __init__(self):
        self.buffer = defaultdict(asyncio.get_event_loop().create_future)

    async def GetValue(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        print()
        value = await self.buffer.get(key)
        return secure_channel_pb2.Value(value=value)

    def AddValueToBuffer(self, request, context):
        key = (request.session_id, request.rendezvous_key)
        self.buffer[key].set_result(request.value)
        return secure_channel_pb2.Empty()


class ChannelServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._servicer = SecureChannelServicer()
        self.server = None

    async def start(self):
        connection = self.host + ":" + self.port
        self.server = aio.server(futures.ThreadPoolExecutor(max_workers=10))
        secure_channel_pb2_grpc.add_SecureChannelServicer_to_server(
            self._servicer, self.server
        )
        self.server.add_insecure_port(connection)
        await self.server.start()
        await self.server.wait_for_termination()

    async def wait(self):
        await self.server.wait_for_termination()
        self.server = None
