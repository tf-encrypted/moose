import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import List

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

    async def execute(self, op, session_id):
        assert isinstance(op, LoadOperation)
        return self.store[op.key]


class SaveKernel(Kernel):
    def __init__(self, store):
        self.store = store

    async def execute(self, op, session_id, value):
        assert isinstance(op, SaveOperation)
        self.store[op.key] = value
        print("Saved {}".format(value))
        return None


class SendKernel(Kernel):
    def __init__(self, channels, delay=None):
        self.channels = channels
        self.delay = delay

    async def execute(self, op, session_id, value):
        assert isinstance(op, SendOperation)
        channel = self.channels[op.channel]
        if self.delay:
            await asyncio.sleep(self.delay)
        await channel.send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )
        return None


class ReceiveKernel(Kernel):
    def __init__(self, channels):
        self.channels = channels

    async def execute(self, op, session_id):
        assert isinstance(op, ReceiveOperation)
        channel = self.channels[op.channel]
        return await channel.receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )


class AddKernel(Kernel):
    async def execute(self, op, session_id, lhs, rhs):
        assert isinstance(op, AddOperation)
        return lhs + rhs


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

    async def run_computation(self, comp, role, session_id):
        session_values = {}
        plan = self.compile_plan(comp, role)
        for op in plan:
            kernel = self.kernels.get(type(op))
            inputs = {
                param_name: session_values[value_name]
                for (param_name, value_name) in op.inputs.items()
            }
            print("{} playing {}: Enter '{}'".format(self.name, role, op.name))
            output = await kernel.execute(op, session_id=session_id, **inputs)
            print("{} playing {}: Exit '{}'".format(self.name, role, op.name))
            if output:
                session_values[op.output] = output

    def compile_plan(self, comp, device_name):
        # TODO(Morten) this is as simple and naive as it gets; we should at least
        # do some kind of topology sorting to make sure we have all async values
        # ready for linking with kernels in `run_computation`
        return [node for node in comp.nodes() if node.device_name == device_name]


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
