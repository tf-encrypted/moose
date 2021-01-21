import grpc
from grpc.experimental import aio as grpc_aio

from moose.asyncstore import AsyncStore
from moose.protos import networking_pb2
from moose.protos import networking_pb2_grpc


class Channel:
    def __init__(self, endpoint, buffer, ca_cert, ident_cert, ident_key):
        self._buffer = buffer

        grpc_aio.init_grpc_aio()
        if ca_cert:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=ident_key,
                certificate_chain=ident_cert,
            )
            self._channel = grpc_aio.secure_channel(endpoint, credentials)
        else:
            self._channel = grpc_aio.insecure_channel(endpoint)

        self._stub = networking_pb2_grpc.NetworkingStub(self._channel)

    async def receive(self, rendezvous_key, session_id):
        reply = await self._stub.GetValue(
            networking_pb2.GetValueRequest(
                rendezvous_key=rendezvous_key, session_id=session_id
            )
        )
        return reply.value

    async def send(self, value, rendezvous_key, session_id):
        key = (session_id, rendezvous_key)
        await self._buffer.put(key, value)


class Networking:
    def __init__(
        self, grpc_server, ca_cert=None, ident_cert=None, ident_key=None,
    ):
        self.buffer = AsyncStore()
        self.channels = dict()
        self.ca_cert = ca_cert
        self.ident_cert = ident_cert
        self.ident_key = ident_key
        networking_pb2_grpc.add_NetworkingServicer_to_server(
            Servicer(self), grpc_server
        )

    def get_hostname(self, placement):
        endpoint = placement
        host, port = endpoint.split(":")
        return host

    async def get_value(self, rendezvous_key, session_id):
        # TODO(Morten) should take caller identity as an argument
        key = (session_id, rendezvous_key)
        return await self.buffer.get(key)

    def get_channel(self, endpoint):
        if endpoint not in self.channels:
            self.channels[endpoint] = Channel(
                endpoint=endpoint,
                buffer=self.buffer,
                ca_cert=self.ca_cert,
                ident_cert=self.ident_cert,
                ident_key=self.ident_key,
            )
        return self.channels[endpoint]

    async def receive(self, sender, receiver, rendezvous_key, session_id):
        return await self.get_channel(sender).receive(
            rendezvous_key=rendezvous_key, session_id=session_id
        )

    async def send(self, value, sender, receiver, rendezvous_key, session_id):
        await self.get_channel(sender).send(
            value, rendezvous_key=rendezvous_key, session_id=session_id
        )


class Servicer(networking_pb2_grpc.NetworkingServicer):
    def __init__(self, networking):
        self.networking = networking

    async def GetValue(self, request, context):
        value = await self.networking.get_value(
            rendezvous_key=request.rendezvous_key, session_id=request.session_id,
        )
        return networking_pb2.GetValueResponse(value=value)
