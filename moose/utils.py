from pathlib import Path

import grpc
from grpc.experimental import aio as grpc_aio

from moose.logger import get_logger


def load_certificate(filename):
    file = Path(filename) if filename else None
    if file and file.exists():
        with open(str(file), "rb") as f:
            cert = f.read()
            return cert
    return None


class DebugInterceptor(grpc_aio.ServerInterceptor):
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
