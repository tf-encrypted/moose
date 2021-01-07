import dill
import numpy as np

from moose.computation.standard import AddOperation
from moose.computation.standard import ConstantOperation
from moose.computation.standard import DeserializeOperation
from moose.computation.standard import DivOperation
from moose.computation.standard import DotOperation
from moose.computation.standard import InputOperation
from moose.computation.standard import InverseOperation
from moose.computation.standard import LoadOperation
from moose.computation.standard import MulOperation
from moose.computation.standard import OnesOperation
from moose.computation.standard import OutputOperation
from moose.computation.standard import ReceiveOperation
from moose.computation.standard import SaveOperation
from moose.computation.standard import SendOperation
from moose.computation.standard import SerializeOperation
from moose.computation.standard import SubOperation
from moose.computation.standard import TransposeOperation
from moose.executor.kernels.base import Kernel
from moose.logger import get_logger


class InputKernel(Kernel):
    async def execute(self, op, session, output):
        assert isinstance(op, InputOperation)
        output.set_result(await session.arguments.get(op.name))


class OutputKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, OutputOperation)
        return None


class ConstantKernel(Kernel):
    def execute_synchronous_block(self, op, session):
        assert isinstance(op, ConstantOperation)
        return op.value


class AddKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, AddOperation)
        return lhs + rhs


class SubKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, SubOperation)
        return lhs - rhs


class MulKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, MulOperation)
        return lhs * rhs


class DotKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, DotOperation)
        return lhs @ rhs


class DivKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, DivOperation)
        return lhs / rhs


class InverseKernel(Kernel):
    def execute_synchronous_block(self, op, session, x):
        assert isinstance(op, InverseOperation)
        assert isinstance(x, np.ndarray)
        return np.linalg.inv(x)


class OnesKernel(Kernel):
    def execute_synchronous_block(self, op, session):
        assert isinstance(op, OnesOperation)
        assert op.dtype in (float, np.float64, int, np.int64)
        return np.ones(shape=op.shape, dtype=op.dtype)


class TransposeKernel(Kernel):
    def execute_synchronous_block(self, op, session, x):
        assert isinstance(op, TransposeOperation)
        assert isinstance(x, np.ndarray)
        return x.transpose(op.axes)


class LoadKernel(Kernel):
    def __init__(self, store):
        self.store = store

    def execute_synchronous_block(self, op, session):
        assert isinstance(op, LoadOperation)
        return self.store[op.key]


class SaveKernel(Kernel):
    def __init__(self, store):
        self.store = store

    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, SaveOperation)
        self.store[op.key] = value
        get_logger().debug(f"Saved {value}")


class SerializeKernel(Kernel):
    async def execute(self, op, session, value, output):
        assert isinstance(op, SerializeOperation)
        value = await value
        value_type = op.value_type
        if value_type == "numpy.ndarray":
            value_ser = dill.dumps(value)
            return output.set_result(value_ser)
        elif value_type == "tf.tensor":
            value_ser = dill.dumps(value)
            return output.set_result(value_ser)
        elif value_type == "tf.keras.model":
            # Model with TF 2.3.0 can't be dilled
            model_json = value.to_json()
            weights = value.get_weights()
            value_ser = dill.dumps((model_json, weights))
            return output.set_result(value_ser)
        else:
            value_ser = dill.dumps(value)
            return output.set_result(value_ser)


class DeserializeKernel(Kernel):
    async def execute(self, op, session, value, output):
        assert isinstance(op, DeserializeOperation)
        value = await value
        value_type = op.value_type
        if value_type == "numpy.array":
            value = dill.loads(value)
            return output.set_result(value)
        elif value_type == "tf.tensor":
            value = dill.loads(value)
            return output.set_result(value)
        elif value_type == "tf.keras.model":
            import tensorflow as tf

            model_json, weights = dill.loads(value)
            model = tf.keras.models.model_from_json(model_json)
            model.set_weights(weights)
            output.set_result(model)
        else:
            value = dill.loads(value)
            output.set_result(value)


class SendKernel(Kernel):
    def __init__(self, networking):
        self.networking = networking

    async def execute(self, op, session, value, output):
        assert isinstance(op, SendOperation)
        await self.networking.send(
            await value,
            sender=session.placement_instantiation.get(op.sender),
            receiver=session.placement_instantiation.get(op.receiver),
            rendezvous_key=op.rendezvous_key,
            session_id=session.session_id,
        )
        output.set_result(None)


class ReceiveKernel(Kernel):
    def __init__(self, networking):
        self.networking = networking

    async def execute(self, op, session, output):
        assert isinstance(op, ReceiveOperation)
        value = await self.networking.receive(
            sender=session.placement_instantiation.get(op.sender),
            receiver=session.placement_instantiation.get(op.receiver),
            rendezvous_key=op.rendezvous_key,
            session_id=session.session_id,
        )
        output.set_result(value)
