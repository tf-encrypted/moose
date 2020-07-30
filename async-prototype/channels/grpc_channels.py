import asyncio
from collections import defaultdict
import grpc
import logging

from grpc.experimental import aio

from channels import secure_channel_pb2
from channels.secure_channel_pb2_grpc import SecureChannelStub


class ChannelManager:
    def __init__(self, channels):
        self.channels = channels

    def get_channel(self, op):
        channel_key = (op.sender, op.receiver)
        return self.channels[channel_key]

    async def send(self, value, op, session_id):
        await self.get_channel(op).send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    async def receive(self, op, session_id):
        return await self.get_channel(op).receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )


class Channel:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._endpoint = self.host + ":" + self.port

    async def receive(self, rendezvous_key, session_id):
        async with aio.insecure_channel(self._endpoint) as channel:
            stub = SecureChannelStub(channel)
            reply = await stub.GetValue(
                secure_channel_pb2.KeyValue(
                    rendezvous_key=rendezvous_key, session_id=session_id
                )
            )
            return reply.value

    async def send(self, value, rendezvous_key, session_id):
        async with aio.insecure_channel(self._endpoint) as channel:
            stub = SecureChannelStub(channel)
            await stub.AddValueToBuffer(
                secure_channel_pb2.RemoteValue(
                    value=value, rendezvous_key=rendezvous_key, session_id=session_id
                )
            )
