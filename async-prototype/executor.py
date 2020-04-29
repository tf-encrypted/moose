import asyncio
from dataclasses import dataclass
from typing import Dict, List

from computation import LoadOperation
from computation import SaveOperation
from computation import SendOperation
from computation import ReceiveOperation
from computation import AddOperation


class Kernel:
    async def execute(self, op, session_id, *values):
        raise NotImplementedError()


class LoadKernel(Kernel):
    def __init__(self, store):
        self.store = store

    async def execute(self, op, session_id):
        assert isinstance(op, LoadOperation)
        return [self.store[op.key]]


class SaveKernel(Kernel):
    def __init__(self, store):
        self.store = store

    async def execute(self, op, session_id, value):
        assert isinstance(op, LoadOperation)
        self.store[op.key] = value
        return []


class SendKernel(Kernel):
    def __init__(self, channels):
        self.channels = channels

    async def execute(self, op, session_id, value):
        assert isinstance(op, SendOperation)
        channel = self.channels[op.channel]
        channel.send(value, rendezvous_key=op.rendezvous_key, session_id=session_id)
        return []


class ReceiveKernel(Kernel):
    def __init__(self, channels):
        self.channels = channels

    async def execute(self, op, session_id):
        assert isinstance(op, ReceiveOperation)
        channel = self.channels[op.channel]
        return [
            channel.receive(rendezvous_key=op.rendezvous_key, session_id=session_id)
        ]


class AddKernel(Kernel):
    async def execute(self, op, session_id, lhs, rhs):
        assert isinstance(op, AddOperation)
        return [lhs + rhs]


class AsyncKernelBasedExecutor:
    def __init__(self, name, store, channels):
        self.name = name
        self.store = store
        self.channels = channels
        self.kernels = {
            LoadOperation: LoadKernel(self.store),
            SaveOperation: SaveKernel(self.store),
            SendOperation: SendKernel(self.channels),
            ReceiveOperation: ReceiveKernel(self.channels),
            AddOperation: AddKernel(),
        }

    def run_computation(self, comp, device_name, session_id):
        session_values = {}
        plan = self.compile_plan(comp, device_name)
        for op in plan:
            kernel = self.kernels.get(type(op))
            inputs = {
                param_name: session_values[value_name]
                for (param_name, value_name) in op.inputs.items()
            }
            outputs = kernel.execute(op, session_id=session_id, **inputs)
            assert isinstance(outputs, list), type(outputs)
            session_values.update(
                {value_name: value for value_name, value in zip(op.outputs, outputs)}
            )

    def compile_plan(self, comp, device_name):
        return [node for node in comp.nodes() if node.device.name == device_name]


class AsyncStore:

    def __init__(self, initial_values: Dict):
        self.values = initial_values

    async def load(self, key):
        return self.values[key]

    async def save(self, key, value):
        self.values[key] = value


class AsyncMemoryChannel:
    def __init__(self):
        self.buffer = {}

    def send(self, value, rendezvous_key, session_id):
        buffer_key = (session_id, rendezvous_key)
        assert buffer_key not in self.buffer
        print("Sending")
        self.buffer[buffer_key] = value

    def receive(self, rendezvous_key, session_id):
        buffer_key = (session_id, rendezvous_key)
        assert buffer_key in self.buffer
        return self.buffer[buffer_key]
