import asyncio
import json
import subprocess
import tempfile

import dill

from moose.compiler.computation import AddOperation
from moose.compiler.computation import CallPythonFunctionOperation
from moose.compiler.computation import ConstantOperation
from moose.compiler.computation import DeserializeOperation
from moose.compiler.computation import DivOperation
from moose.compiler.computation import LoadOperation
from moose.compiler.computation import MulOperation
from moose.compiler.computation import ReceiveOperation
from moose.compiler.computation import RunProgramOperation
from moose.compiler.computation import SaveOperation
from moose.compiler.computation import SendOperation
from moose.compiler.computation import SerializeOperation
from moose.compiler.computation import SubOperation
from moose.executor.kernels.base import Kernel
from moose.logger import get_logger


class AddKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, AddOperation)
        return lhs + rhs


class CallPythonFunctionKernel(Kernel):
    async def execute(self, op, session, output, **inputs):
        assert isinstance(op, CallPythonFunctionOperation)
        python_fn = dill.loads(op.pickled_fn)
        concrete_inputs = await asyncio.gather(*inputs.values())
        concrete_output = python_fn(*concrete_inputs)
        output.set_result(concrete_output)


class ConstantKernel(Kernel):
    def execute_synchronous_block(self, op, session):
        assert isinstance(op, ConstantOperation)
        return op.value


class DeserializeKernel(Kernel):
    async def execute(self, op, session, value, output=None):
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


class DivKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, DivOperation)
        return lhs / rhs


class LoadKernel(Kernel):
    def __init__(self, store):
        self.store = store

    def execute_synchronous_block(self, op, session):
        assert isinstance(op, LoadOperation)
        return self.store[op.key]


class MulKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, MulOperation)
        return lhs * rhs


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


class RunProgramKernel(Kernel):
    async def execute(self, op, session, output, **inputs):
        assert isinstance(op, RunProgramOperation)
        with tempfile.NamedTemporaryFile() as inputfile:
            with tempfile.NamedTemporaryFile() as outputfile:

                concrete_inputs = await asyncio.gather(*inputs.values())
                inputfile.write(json.dumps(concrete_inputs).encode())
                inputfile.flush()

                args = [
                    op.path,
                    *op.args,
                    "--input-file",
                    inputfile.name,
                    "--output-file",
                    outputfile.name,
                    "--session-id",
                    str(session.session_id),
                    "--placement",
                    op.placement_name,
                ]
                get_logger().debug(f"Running external program: {args}")
                _ = subprocess.run(
                    args, stdout=subprocess.PIPE, universal_newlines=True,
                )

                concrete_output = json.loads(outputfile.read())

        output.set_result(concrete_output)


class SaveKernel(Kernel):
    def __init__(self, store):
        self.store = store

    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, SaveOperation)
        self.store[op.key] = value
        get_logger().debug(f"Saved {value}")


class SerializeKernel(Kernel):
    async def execute(self, op, session, value, output=None):
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


class SendKernel(Kernel):
    def __init__(self, networking):
        self.networking = networking

    async def execute(self, op, session, value, output=None):
        assert isinstance(op, SendOperation)
        await self.networking.send(
            await value,
            sender=session.placement_instantiation.get(op.sender),
            receiver=session.placement_instantiation.get(op.receiver),
            rendezvous_key=op.rendezvous_key,
            session_id=session.session_id,
        )


class SubKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, SubOperation)
        return lhs - rhs
