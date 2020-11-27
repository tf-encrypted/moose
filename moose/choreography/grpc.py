import asyncio
import random
from typing import Dict

import grpc
from grpc.experimental import aio as grpc_aio

from moose.compiler.base import Computation
from moose.logger import get_logger
from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc


class Choreographer:
    def __init__(self, ca_cert=None, ident_cert=None, ident_key=None,) -> None:
        self.ca_cert = ca_cert
        self.ident_cert = ident_cert
        self.ident_key = ident_key
        self.existing_executors = dict()

    def evaluate_computation(
        self, computation: Computation, placement_instantiation: Dict
    ):
        placement_instantiation = {
            placement.name if not isinstance(placement, str) else placement: endpoint
            for placement, endpoint in placement_instantiation.items()
        }
        placement_executors = dict()
        for placement, endpoint in placement_instantiation.items():
            if endpoint not in self.existing_executors:
                self.existing_executors[endpoint] = ExecutorProxy(
                    endpoint,
                    ca_cert=self.ca_cert,
                    ident_cert=self.ident_cert,
                    ident_key=self.ident_key,
                )
            placement_executors[placement] = self.existing_executors[endpoint]

        sid = random.randrange(2 ** 32)
        tasks = [
            executor.run_computation(
                computation,
                placement_instantiation=placement_instantiation,
                placement=placement,
                session_id=sid,
            )
            for placement, executor in placement_executors.items()
        ]
        joint_task = asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        done, _ = asyncio.get_event_loop().run_until_complete(joint_task)
        exceptions = [task.exception() for task in done if task.exception()]
        for e in exceptions:
            get_logger().exception(e)
        if exceptions:
            raise Exception(
                "One or more errors evaluting the computation, see log for details"
            )


class ExecutorProxy:
    def __init__(self, endpoint, ca_cert, ident_cert, ident_key):
        grpc_aio.init_grpc_aio()
        if ca_cert:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=ident_key,
                certificate_chain=ident_cert,
            )
            self.channel = grpc_aio.secure_channel(endpoint, credentials)
        else:
            self.channel = grpc_aio.insecure_channel(endpoint)
        self._stub = executor_pb2_grpc.ExecutorStub(self.channel)

    async def run_computation(
        self, logical_computation, placement_instantiation, placement, session_id
    ):
        comp_ser = logical_computation.serialize()
        compute_request = executor_pb2.RunComputationRequest(
            computation=comp_ser,
            placement_instantiation=placement_instantiation,
            placement=placement,
            session_id=session_id,
        )
        _ = await self._stub.RunComputation(compute_request)


class Choreography:
    def __init__(
        self, executor, grpc_server,
    ):
        executor_pb2_grpc.add_ExecutorServicer_to_server(
            Servicer(executor), grpc_server
        )


class Servicer(executor_pb2_grpc.ExecutorServicer):
    def __init__(self, executor):
        self.executor = executor

    async def RunComputation(self, request, context):
        await self.executor.run_computation(
            logical_computation=Computation.deserialize(request.computation),
            placement_instantiation=request.placement_instantiation,
            placement=request.placement,
            session_id=request.session_id,
        )
        return executor_pb2.RunComputationResponse()
