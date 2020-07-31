import asyncio
from collections import defaultdict
import grpc
import logging

from grpc.experimental import aio

from channels.gen import secure_channel_pb2
from channels.gen.secure_channel_pb2_grpc import SecureChannelStub


class ChannelManager:
    def __init__(self, cluster_spec):
        self.channels = self.create_channels(cluster_spec)

    def get_channel(self, op):
        channel_key = op.sender
        return self.channels[channel_key]

    async def send(self, value, op, session_id):
        await self.get_channel(op).send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    async def receive(self, op, session_id):
        return await self.get_channel(op).receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    def create_channels(self, cluster_spec):
        channels = {}
        for player in cluster_spec:
            host, port = cluster_spec[player].split(':')
            channels[player] = Channel(host, port)
        return channels


class Channel:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._endpoint = self.host + ":" + self.port
        self.channel_recv = aio.insecure_channel(self._endpoint)
        self.channel_send = aio.insecure_channel(self._endpoint)

    async def receive(self, rendezvous_key, session_id):
        async with self.channel_recv:
            stub = SecureChannelStub(self.channel_recv)
            reply = await stub.GetValue(
                secure_channel_pb2.KeyValue(
                    rendezvous_key=rendezvous_key, session_id=session_id
                )
            )
            return reply.value

    async def send(self, value, rendezvous_key, session_id):
        async with self.channel_send:
            stub = SecureChannelStub(self.channel_send)
            await stub.AddValueToBuffer(
                secure_channel_pb2.RemoteValue(
                    value=value, rendezvous_key=rendezvous_key, session_id=session_id
                )
            )
