import grpc
from grpc.experimental import aio

from moose.logger import get_logger
from moose.protos import broker_pb2
from moose.protos import broker_pb2_grpc
from moose.utils import load_certificate


class BrokerServicer(board_pb2_grpc.BoardServicer):
    def __init__(self, broker):
        self.broker = broker

    async def Pub(self, request_iterator, context):
        for request in request_iterator:
            receiver = request.receiver
            session_id = request.session_id
            rendezvous_key = request.rendezvous_key
            value = request.value
            message = ()
            queue = self.broker.pub(
                subject=f"{request.receiver}/{request.session_id}/{request.rendezvous_key}",
                value=request.value,
            )
            get_logger().debug(f"{request.session_id}")

    async def GetValue(self, request, context):
        receiver = request.receiver
        queue = self.queues.get(receiver, None)
        assert queue
        while True:
            message = await queue.get()
            yield GetValueResponse(message)


class Broker:
    def __init__(
        self,
        host,
        port,
        ca_cert_filename=None,
        ident_cert_filename=None,
        ident_key_filename=None,
        allow_insecure_networking=False,
    ):
        self.server = self.setup_server(
            host=host,
            port=port,
            ca_cert_filename=ca_cert_filename,
            ident_cert_filename=ident_cert_filename,
            ident_key_filename=ident_key_filename,
            allow_insecure_networking=allow_insecure_networking,
        )

    def setup_server(
        self,
        host,
        port,
        ca_cert_filename,
        ident_cert_filename,
        ident_key_filename,
        allow_insecure_networking,
    ):
        ca_cert = load_certificate(ca_cert_filename)
        ident_cert = load_certificate(ident_cert_filename)
        ident_key = load_certificate(ident_key_filename)

        aio.init_grpc_aio()
        server = aio.server()

        if ident_cert and ident_key:
            get_logger().info(f"Setting up server at {host}:{port}")
            credentials = grpc.ssl_server_credentials(
                [(ident_key, ident_cert)],
                root_certificates=ca_cert,
                require_client_auth=True,
            )
            server.add_secure_port(f"{host}:{port}", credentials)
        else:
            assert allow_insecure_networking
            get_logger().warning(
                f"Setting up server at {host}:{port} with insecure networking"
            )
            server.add_insecure_port(f"{host}:{port}")

        board_pb2_grpc.add_BrokerServicer_to_server(BrokerServicer(self), server)
        return server

    async def start(self):
        await self._server.start()

    async def wait(self):
        await self._server.wait_for_termination()

    async def pub(self, receiver, message):
        pass
