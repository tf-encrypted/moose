import grpc
from grpc.experimental import aio

from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc


class RemoteExecutor:
    def __init__(self, endpoint, ca_cert, ident_cert, ident_key):
        aio.init_grpc_aio()
        if ca_cert:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=ident_key,
                certificate_chain=ident_cert,
            )
            self.channel = aio.secure_channel(endpoint, credentials)
        else:
            self.channel = aio.insecure_channel(endpoint)
        self._stub = executor_pb2_grpc.ExecutorStub(self.channel)

    async def run_computation(
        self, logical_computation, placement, placement_instantiation, session_id
    ):
        comp_ser = logical_computation.serialize()
        compute_request = executor_pb2.RunComputationRequest(
            computation=comp_ser,
            placement=placement,
            placement_instantiation=placement_instantiation,
            session_id=session_id,
        )
        _ = await self._stub.RunComputation(compute_request)
