import asyncio
from collections import defaultdict

from computation import AddOperation
from computation import LoadOperation
from computation import ReceiveOperation
from computation import SaveOperation
from computation import SendOperation


class Kernel:
    async def execute(self, op, session_id, *values):
        raise NotImplementedError()


class LoadKernel(Kernel):
    def __init__(self, store):
        self.store = store

    async def execute(self, op, session_id, output):
        assert isinstance(op, LoadOperation)
        value = self.store[op.key]
        print(f"Loaded {value}")
        output.set_result(value)


class SaveKernel(Kernel):
    def __init__(self, store):
        self.store = store

    async def execute(self, op, session_id, value, output=None):
        assert isinstance(op, SaveOperation)
        self.store[op.key] = await value
        print(f"Saved {await value}")


class SendKernel(Kernel):
    def __init__(self, channels, delay=None):
        self.channels = channels
        self.delay = delay

    async def execute(self, op, session_id, value, output=None):
        assert isinstance(op, SendOperation)
        channel = self.channels[op.channel]
        if self.delay:
            await asyncio.sleep(self.delay)
        value = await value
        await channel.send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )


class ReceiveKernel(Kernel):
    def __init__(self, channels):
        self.channels = channels

    async def execute(self, op, session_id, output):
        assert isinstance(op, ReceiveOperation)
        channel = self.channels[op.channel]
        value = await channel.receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )
        output.set_result(value)


class AddKernel(Kernel):
    async def execute(self, op, session_id, lhs, rhs, output):
        assert isinstance(op, AddOperation)
        res = (await lhs) + (await rhs)
        output.set_result(res)


class AsyncKernelBasedExecutor:
    def __init__(self, name, store, channels, send_delay=None):
        self.name = name
        self.store = store
        self.channels = channels
        self.kernels = {
            LoadOperation: LoadKernel(self.store),
            SaveOperation: SaveKernel(self.store),
            SendOperation: SendKernel(self.channels, delay=send_delay),
            ReceiveOperation: ReceiveKernel(self.channels),
            AddOperation: AddKernel(),
        }

    async def run_computation(self, logical_computation, role, session_id, event_loop):
        physical_computation = self.compile_computation(logical_computation)
        execution_plan = self.schedule_execution(physical_computation, role)
        # create futures for all edges in the graph
        # - note that this could be done lazily as well
        session_values = {op.output: event_loop.create_future() for op in execution_plan if op.output}
        # link futures together using kernels
        tasks = []
        for op in execution_plan:
            kernel = self.kernels.get(type(op))
            inputs = {
                param_name: session_values[value_name]
                for (param_name, value_name) in op.inputs.items()
            }
            output = session_values[op.output] if op.output else None
            print("{} playing {}: Enter '{}'".format(self.name, role, op.name))
            kernel_task = kernel.execute(op, session_id=session_id, output=output, **inputs)
            tasks += [kernel_task]
            print("{} playing {}: Exit '{}'".format(self.name, role, op.name))
        await asyncio.gather(*tasks)

    def compile_computation(self, logical_computation):
        # TODO for now we don't do any compilation of computations
        return logical_computation

    def schedule_execution(self, comp, role):
        # TODO(Morten) this is as simple and naive as it gets; we should at least
        # do some kind of topology sorting to make sure we have all async values
        # ready for linking with kernels in `run_computation`
        return [node for node in comp.nodes() if node.device_name == role]


class AsyncStore:
    def __init__(self, initial_values):
        self.values = initial_values

    async def load(self, key):
        return self.values[key]

    async def save(self, key, value):
        self.values[key] = value


class AsyncMemoryChannel:
    def __init__(self):
        # TODO(Morten) having an async dict-like structure
        # would probably be better ...
        self.queues = defaultdict(asyncio.Queue)

    async def send(self, value, rendezvous_key, session_id):
        queue_key = (session_id, rendezvous_key)
        queue = self.queues[queue_key]
        return await queue.put(value)

    async def receive(self, rendezvous_key, session_id):
        queue_key = (session_id, rendezvous_key)
        queue = self.queues[queue_key]
        return await queue.get()
