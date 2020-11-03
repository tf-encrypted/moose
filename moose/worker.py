import grpc
from grpc.experimental import aio

from moose.cluster.cluster_spec import load_cluster_spec
from moose.compiler.computation import Computation
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.grpc_direct import ChannelManager
from moose.protos import channel_manager_pb2
from moose.protos import channel_manager_pb2_grpc
from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc
from moose.utils import load_certificate


class ExecutorServicer(executor_pb2_grpc.ExecutorServicer):
    def __init__(self, executor):
        self.executor = executor

    async def RunComputation(self, request, context):
        await self.executor.run_computation(
            logical_computation=Computation.deserialize(request.computation),
            placement=request.placement,
            session_id=request.session_id,
        )
        return executor_pb2.RunComputationResponse()


class ChannelManagerServicer(channel_manager_pb2_grpc.ChannelManagerServicer):
    def __init__(self, channel_manager):
        self.channel_manager = channel_manager

    async def GetValue(self, request, context):
        value = await self.channel_manager.get_value(
            rendezvous_key=request.rendezvous_key, session_id=request.session_id,
        )
        return channel_manager_pb2.GetValueResponse(value=value)


class DebugInterceptor(aio.ServerInterceptor):
    def __init__(self):
        self.handler_type = {
            (False, False): grpc.unary_unary_rpc_method_handler,
        }

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)

        async def intercepted_handler(request, context):
            get_logger().debug(
                f"Incoming gRPC, "
                f"method:'{handler_call_details.method}', "
                f"peer:'{context.peer()}', "
                f"peer_identities:'{context.peer_identities()}'"
            )
            return await handler.unary_unary(request, context)

        handler_type = self.handler_type.get(
            (handler.request_streaming, handler.response_streaming), None
        )
        if not handler_type:
            raise NotImplementedError(f"Unknown handler {handler}")
        return handler_type(
            intercepted_handler,
            handler.request_deserializer,
            handler.response_serializer,
        )


class Worker:
    def __init__(
        self,
        name,
        host,
        port,
        cluster_spec_filename,
        ca_cert_filename=None,
        ident_cert_filename=None,
        ident_key_filename=None,
        allow_insecure_networking=False,
    ):
        ca_cert = load_certificate(ca_cert_filename)
        ident_cert = load_certificate(ident_cert_filename)
        ident_key = load_certificate(ident_key_filename)

        cluster_spec = load_cluster_spec(cluster_spec_filename)

        channel_manager = ChannelManager(
            cluster_spec, ca_cert=ca_cert, ident_cert=ident_cert, ident_key=ident_key
        )
        executor = AsyncExecutor(name=name, channel_manager=channel_manager)

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

        executor_pb2_grpc.add_ExecutorServicer_to_server(
            ExecutorServicer(executor), self._server,
        )

        channel_manager_pb2_grpc.add_ChannelManagerServicer_to_server(
            ChannelManagerServicer(channel_manager), self._server,
        )

    async def start(self):
        await self._server.start()

    async def wait(self):
        await self._server.wait_for_termination()
