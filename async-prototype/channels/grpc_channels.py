import asyncio
import logging
from collections import defaultdict

import grpc
from grpc.experimental import aio

from protos import executor_pb2
from protos import executor_pb2_grpc


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
            host, port = cluster_spec[player].split(":")
            channels[player] = Channel(host, port)
        return channels


class Channel:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._endpoint = self.host + ":" + self.port
        self.channel = aio.insecure_channel(self._endpoint)
        self._stub = executor_pb2_grpc.ExecutorStub(self.channel)

    async def receive(self, rendezvous_key, session_id):
        reply = await self._stub.GetValue(
            executor_pb2.ValueRequest(
                rendezvous_key=rendezvous_key, session_id=session_id
            )
        )
        return reply.value

    async def send(self, value, rendezvous_key, session_id):
        await self._stub.AddValueToBuffer(
            executor_pb2.RemoteValue(
                value=value, rendezvous_key=rendezvous_key, session_id=session_id
            )
        )
