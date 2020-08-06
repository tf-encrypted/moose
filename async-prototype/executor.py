import asyncio
from collections import defaultdict
from grpc.experimental import aio

from computation import AddOperation
from computation import ConstantOperation
from computation import LoadOperation
from computation import MulOperation
from computation import ReceiveOperation
from computation import SaveOperation
from computation import SendOperation
from computation import SubOperation
from protos import executor_pb2
from protos import executor_pb2_grpc

from logger import get_logger


class Kernel:
    async def execute(self, op, session_id, **kwargs):
        raise NotImplementedError()


class StrictKernel(Kernel):
    async def execute(self, op, session_id, output, **kwargs):
        concrete_kwargs = {key: await value for key, value in kwargs.items()}
        concrete_output = self.strict_execute(
            op=op, session_id=session_id, **concrete_kwargs
        )
        if output:
            output.set_result(concrete_output)

    def strict_execute(self, op, session_id, **kwargs):
        raise NotImplementedError()


class LoadKernel(StrictKernel):
    def __init__(self, store):
        self.store = store

    def strict_execute(self, op, session_id):
        assert isinstance(op, LoadOperation)
        return self.store[op.key]


class SaveKernel(StrictKernel):
    def __init__(self, store):
        self.store = store

    def strict_execute(self, op, session_id, value):
        assert isinstance(op, SaveOperation)
        self.store[op.key] = value
        get_logger().debug(f"Saved {value}")


class SendKernel(Kernel):
    def __init__(self, channel_manager):
        self.channel_manager = channel_manager

    async def execute(self, op, session_id, value, output=None):
        assert isinstance(op, SendOperation)
        await self.channel_manager.send(await value, op=op, session_id=session_id)


class ReceiveKernel(Kernel):
    def __init__(self, channel_manager):
        self.channel_manager = channel_manager

    async def execute(self, op, session_id, output):
        assert isinstance(op, ReceiveOperation)
        value = await self.channel_manager.receive(op=op, session_id=session_id)
        output.set_result(value)


class ConstantKernel(StrictKernel):
    async def execute(self, op, session_id, output):
        print("constant")
        assert isinstance(op, ConstantOperation)
        return output.set_result(op.value)


class AddKernel(StrictKernel):
    def strict_execute(self, op, session_id, lhs, rhs):
        assert isinstance(op, AddOperation)
        return lhs + rhs


class SubKernel(StrictKernel):
    def strict_execute(self, op, session_id, lhs, rhs):
        assert isinstance(op, SubOperation)
        return lhs - rhs


class MulKernel(StrictKernel):
    def strict_execute(self, op, session_id, lhs, rhs):
        assert isinstance(op, MulOperation)
        return lhs * rhs


class AsyncKernelBasedExecutor:
    def __init__(self, name, channel_manager, store={}):
        self.name = name
        self.kernels = {
            LoadOperation: LoadKernel(store),
            SaveOperation: SaveKernel(store),
            SendOperation: SendKernel(channel_manager),
            ReceiveOperation: ReceiveKernel(channel_manager),
            ConstantOperation: ConstantKernel(),
            AddOperation: AddKernel(),
            SubOperation: SubKernel(),
            MulOperation: MulKernel(),
        }

    async def run_computation(self, logical_computation, role, session_id, event_loop):
        physical_computation = self.compile_computation(logical_computation)
        execution_plan = self.schedule_execution(physical_computation, role)
        # lazily create futures for all edges in the graph
        session_values = defaultdict(event_loop.create_future)
        # link futures together using kernels
        tasks = []
        for op in execution_plan:
            kernel = self.kernels.get(type(op))
            if not kernel:
                get_logger().fatal(f"No kernel found for operation {type(op)}")
            inputs = {
                param_name: session_values[value_name]
                for (param_name, value_name) in op.inputs.items()
            }
            output = session_values[op.output] if op.output else None
            get_logger().debug(f"{self.name} playing {role}: Enter '{op.name}'")
            tasks += [
                asyncio.create_task(
                    kernel.execute(op, session_id=session_id, output=output, **inputs)
                )
            ]
            get_logger().debug(f"{self.name} playing {role}: Exit '{op.name}'")
        await asyncio.wait(tasks)

    def compile_computation(self, logical_computation):
        # TODO for now we don't do any compilation of computations
        return logical_computation

    def schedule_execution(self, comp, role):
        # TODO(Morten) this is as simple and naive as it gets; we should at least
        # do some kind of topology sorting to make sure we have all async values
        # ready for linking with kernels in `run_computation`
        return [node for node in comp.nodes() if node.device_name == role]


class RemoteExecutor:
    def __init__(self, endpoint):
        self.channel = aio.insecure_channel(endpoint)
        self._stub = executor_pb2_grpc.ExecutorStub(self.channel)

    async def run_computation(self, logical_computation, role, session_id):
        compute_request = executor_pb2.ComputeRequest(
            computation=logical_computation, role=role, session_id=session_id
        )
        response = await self._stub.RunComputation(compute_request)


class AsyncStore:
    def __init__(self, initial_values):
        self.values = initial_values

    async def load(self, key):
        return self.values[key]

    async def save(self, key, value):
        self.values[key] = value
