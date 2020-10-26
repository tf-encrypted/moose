from grpc.experimental import aio

from moose.protos import channel_manager_pb2
from moose.protos import channel_manager_pb2_grpc
from moose.storage import AsyncStore


class Channel:
    def __init__(self, endpoint, buffer):
        self._buffer = buffer
        self._channel = aio.insecure_channel(endpoint)
        self._stub = channel_manager_pb2_grpc.ChannelManagerStub(self._channel)

    async def receive(self, rendezvous_key, session_id):
        reply = await self._stub.GetValue(
            channel_manager_pb2.GetValueRequest(
                rendezvous_key=rendezvous_key, session_id=session_id
            )
        )
        return reply.value

    async def send(self, value, rendezvous_key, session_id):
        key = (session_id, rendezvous_key)
        await self._buffer.put(key, value)


class ChannelManager:
    def __init__(self, cluster_spec):
        self.buffer = AsyncStore()
        self.endpoints = {player: endpoint for player, endpoint in cluster_spec.items()}
        self.channels = {
            player: Channel(endpoint, self.buffer)
            for player, endpoint in cluster_spec.items()
        }

    def get_hostname(self, player_name):
        endpoint = self.endpoints.get(player_name)
        host, port = endpoint.split(":")
        return host

    def get_channel(self, op):
        return self.channels[op.sender]

    async def get_value(self, rendezvous_key, session_id):
        # TODO(Morten) should take caller identity as an argument
        key = (session_id, rendezvous_key)
        return await self.buffer.get(key)

    async def receive(self, op, session_id):
        return await self.get_channel(op).receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    async def send(self, value, op, session_id):
        await self.get_channel(op).send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )
