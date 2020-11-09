import grpc
from grpc.experimental import aio

from moose.choreography.grpc import ExecutorServicer
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.grpc import Networking
from moose.networking.grpc import NetworkingServicer
from moose.utils import DebugInterceptor
from moose.utils import load_certificate


class Worker:
    def __init__(
        self,
        name,
        host,
        port,
        ca_cert_filename=None,
        ident_cert_filename=None,
        ident_key_filename=None,
        allow_insecure_networking=False,
    ):
        ca_cert = load_certificate(ca_cert_filename)
        ident_cert = load_certificate(ident_cert_filename)
        ident_key = load_certificate(ident_key_filename)

        # set up server
        aio.init_grpc_aio()
        self._server = aio.server(interceptors=(DebugInterceptor(),))

        if ident_cert and ident_key:
            get_logger().info(f"Setting up server at {host}:{port}")
            credentials = grpc.ssl_server_credentials(
                [(ident_key, ident_cert)],
                root_certificates=ca_cert,
                require_client_auth=True,
            )
            self._server.add_secure_port(f"{host}:{port}", credentials)
        else:
            assert allow_insecure_networking
            get_logger().warning(
                f"Setting up server at {host}:{port} with insecure networking"
            )
            self._server.add_insecure_port(f"{host}:{port}")

        networking = Networking(
            ca_cert=ca_cert, ident_cert=ident_cert, ident_key=ident_key
        )
        networking_servicer = NetworkingServicer(networking)
        networking_servicer.add_to_server(self._server)

        executor = AsyncExecutor(name=name, networking=networking)
        executor_servicer = ExecutorServicer(executor)
        executor_servicer.add_to_server(self._server)

    async def start(self):
        await self._server.start()

    async def wait(self):
        await self._server.wait_for_termination()
