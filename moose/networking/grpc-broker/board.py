import grpc
from grpc.experimental import aio

from moose.logger import get_logger
from moose.protos import board_pb2
from moose.protos import board_pb2_grpc
from moose.storage import AsyncStore
from moose.utils import load_certificate


class Channel:
    def __init__(self, endpoint, buffer, ca_cert, ident_cert, ident_key):
        self._buffer = buffer

        
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


class NetworkManager:
    def __init__(self, ca_cert, ident_cert, ident_key):
        aio.init_grpc_aio()
        self.client = self.setup_client(ca_cert=ca_cert, ident_cert=ident_cert, ident_key=ident_key,)

    def setup_client(self, ca_cert, ident_cert, ident_key):
        if ca_cert:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=ident_key,
                certificate_chain=ident_cert,
            )
            channel = aio.secure_channel(endpoint, credentials)
        else:
            channel = aio.insecure_channel(endpoint)

        client = channel_manager_pb2_grpc.ChannelManagerStub(self._channel)
        return client

    def get_hostname(self, player_name):
        raise NotImplementedError()

    async def get_value(self, rendezvous_key, session_id):
        # TODO(Morten) should take caller identity as an argument
        key = (session_id, rendezvous_key)
        return await self.buffer.get(key)

    async def receive(self, op, session_id):
        return await self.get_channel(op).receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    async def send(self, value, op, session_id):
        subject = op.receiver
        

        await self.get_channel(op).send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )
