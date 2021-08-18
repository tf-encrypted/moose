import grpc
from grpc.experimental import aio as grpc_aio

from moose.deprecated.choreography.grpc import Choreography
from moose.deprecated.utils import DebugInterceptor
from moose.deprecated.utils import load_certificate
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.grpc import Networking
from moose.storage.memory import MemoryDataStore


class Worker:
    def __init__(
        self,
        port,
        host="0.0.0.0",
        ca_cert_filename=None,
        ident_cert_filename=None,
        ident_key_filename=None,
        allow_insecure_networking=False,
    ):
        ca_cert = load_certificate(ca_cert_filename)
        ident_cert = load_certificate(ident_cert_filename)
        ident_key = load_certificate(ident_key_filename)

        self.grpc_server = self.setup_server(
            port=port,
            host=host,
            ca_cert=ca_cert,
            ident_cert=ident_cert,
            ident_key=ident_key,
            allow_insecure_networking=allow_insecure_networking,
        )
        networking = Networking(
            grpc_server=self.grpc_server,
            ca_cert=ca_cert,
            ident_cert=ident_cert,
            ident_key=ident_key,
        )
        storage = MemoryDataStore()
        executor = AsyncExecutor(networking=networking, storage=storage)
        self.choreography = Choreography(
            executor=executor, grpc_server=self.grpc_server,
        )

    def setup_server(
        self,
        port,
        host,
        ca_cert,
        ident_cert,
        ident_key,
        allow_insecure_networking,
        debug=False,
    ):
        grpc_aio.init_grpc_aio()
        if debug:
            grpc_server = grpc_aio.server(interceptors=(DebugInterceptor(),))
        else:
            grpc_server = grpc_aio.server()

        if ident_cert and ident_key:
            get_logger().debug(f"Setting up secure server at {host}:{port}")
            credentials = grpc.ssl_server_credentials(
                [(ident_key, ident_cert)],
                root_certificates=ca_cert,
                require_client_auth=True,
            )
            grpc_server.add_secure_port(f"{host}:{port}", credentials)
        else:
            assert allow_insecure_networking
            get_logger().warning(f"Setting up insecure server at {host}:{port}")
            grpc_server.add_insecure_port(f"{host}:{port}")

        return grpc_server

    async def start(self):
        await self.grpc_server.start()

    async def wait_for_termination(self):
        await self.grpc_server.wait_for_termination()
