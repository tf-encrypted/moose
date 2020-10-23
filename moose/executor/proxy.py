from grpc.experimental import aio

from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc


class RemoteExecutor:
    def __init__(self, endpoint):
        self.channel = aio.insecure_channel(endpoint)
        self._stub = executor_pb2_grpc.ExecutorStub(self.channel)

    async def run_computation(self, logical_computation, placement, session_id):
        comp_ser = logical_computation.serialize()
        compute_request = executor_pb2.RunComputationRequest(
            computation=comp_ser, placement=placement, session_id=session_id
        )
        _ = await self._stub.RunComputation(compute_request)
