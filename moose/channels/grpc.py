from grpc.experimental import aio

from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc


class Channel:
    def __init__(self, endpoint):
        self._channel = aio.insecure_channel(endpoint)
        self._stub = executor_pb2_grpc.ExecutorStub(self._channel)

    async def receive(self, rendezvous_key, session_id):
        reply = await self._stub.GetValue(
            executor_pb2.GetValueRequest(
                rendezvous_key=rendezvous_key, session_id=session_id
            )
        )
        return reply.value

    async def send(self, value, rendezvous_key, session_id):
        await self._stub.SetValue(
            executor_pb2.SetValueRequest(
                value=value, rendezvous_key=rendezvous_key, session_id=session_id
            )
        )


class ChannelManager:
    def __init__(self, cluster_spec):
        self.endpoints = {
            player: endpoint for player, endpoint in cluster_spec.items()
        }
        self.channels = {
            player: Channel(endpoint) for player, endpoint in cluster_spec.items()
        }

    def get_hostname(self, player_name):
        endpoint = self.endpoints.get(player_name)
        host, port = endpoint.split(":")
        return host

    def get_channel(self, op):
        return self.channels[op.sender]

    async def receive(self, op, session_id):
        return await self.get_channel(op).receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    async def send(self, value, op, session_id):
        await self.get_channel(op).send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )
