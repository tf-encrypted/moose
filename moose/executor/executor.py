import asyncio
import json
import os
import pathlib
import subprocess
import tempfile

import dill
from grpc.experimental import aio

from moose.compiler.computation import AddOperation
from moose.compiler.computation import CallPythonFunctionOperation
from moose.compiler.computation import ConstantOperation
from moose.compiler.computation import DeserializeOperation
from moose.compiler.computation import DivOperation
from moose.compiler.computation import LoadOperation
from moose.compiler.computation import MpspdzCallOperation
from moose.compiler.computation import MpspdzLoadOutputOperation
from moose.compiler.computation import MpspdzSaveInputOperation
from moose.compiler.computation import MulOperation
from moose.compiler.computation import ReceiveOperation
from moose.compiler.computation import RunProgramOperation
from moose.compiler.computation import SaveOperation
from moose.compiler.computation import SendOperation
from moose.compiler.computation import SerializeOperation
from moose.compiler.computation import SubOperation
from moose.logger import get_logger
from moose.protos import executor_pb2
from moose.protos import executor_pb2_grpc
from moose.storage import AsyncStore


class Kernel:
    async def execute(self, op, session_id, output, **kwargs):
        concrete_kwargs = {key: await value for key, value in kwargs.items()}
        concrete_output = self.execute_synchronous_block(
            op=op, session_id=session_id, **concrete_kwargs
        )
        if output:
            output.set_result(concrete_output)

    def execute_synchronous_block(self, op, session_id, **kwargs):
        raise NotImplementedError()


class AddKernel(Kernel):
    def execute_synchronous_block(self, op, session_id, lhs, rhs):
        assert isinstance(op, AddOperation)
        return lhs + rhs


class CallPythonFunctionKernel(Kernel):
    async def execute(self, op, session_id, output, **inputs):
        python_fn = dill.loads(op.pickled_fn)
        concrete_inputs = await asyncio.gather(*inputs.values())
        concrete_output = python_fn(*concrete_inputs)
        output.set_result(concrete_output)


class ConstantKernel(Kernel):
    def execute_synchronous_block(self, op, session_id):
        assert isinstance(op, ConstantOperation)
        return op.value


class DeserializeKernel(Kernel):
    async def execute(self, op, session_id, value, output=None):
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
    def execute_synchronous_block(self, op, session_id, lhs, rhs):
        assert isinstance(op, DivOperation)
        return lhs / rhs


class LoadKernel(Kernel):
    def __init__(self, store):
        self.store = store

    def execute_synchronous_block(self, op, session_id):
        assert isinstance(op, LoadOperation)
        return self.store[op.key]


class MulKernel(Kernel):
    def execute_synchronous_block(self, op, session_id, lhs, rhs):
        assert isinstance(op, MulOperation)
        return lhs * rhs


class ReceiveKernel(Kernel):
    def __init__(self, channel_manager):
        self.channel_manager = channel_manager

    async def execute(self, op, session_id, output):
        assert isinstance(op, ReceiveOperation)
        value = await self.channel_manager.receive(op=op, session_id=session_id)
        output.set_result(value)


class RunProgramKernel(Kernel):
    async def execute(self, op, session_id, output, **inputs):
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
                    str(session_id),
                    "--device",
                    op.device_name,
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

    def execute_synchronous_block(self, op, session_id, value):
        assert isinstance(op, SaveOperation)
        self.store[op.key] = value
        get_logger().debug(f"Saved {value}")


class SerializeKernel(Kernel):
    async def execute(self, op, session_id, value, output=None):
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
    def __init__(self, channel_manager):
        self.channel_manager = channel_manager

    async def execute(self, op, session_id, value, output=None):
        assert isinstance(op, SendOperation)
        await self.channel_manager.send(await value, op=op, session_id=session_id)


class SubKernel(Kernel):
    def execute_synchronous_block(self, op, session_id, lhs, rhs):
        assert isinstance(op, SubOperation)
        return lhs - rhs


class MpspdzSaveInputKernel(Kernel):
    def execute_synchronous_block(self, op, session_id, **inputs):
        assert isinstance(op, MpspdzSaveInputOperation)
        # Player-Data/Input-P0-0
        output = subprocess.call(
            [
                "./mpspdz-links.sh",
                f"{session_id}",
                f"{op.invocation_key}",
            ]
        )
        mpspdz_dir = (
            f"/MP-SPDZ/tmp/{session_id}/{op.invocation_key}/Player-Data/Input-P"
        )
        thread_no = 0  # assume inputs are happening in the main thread
        mpspdz_input_file = f"{mpspdz_dir}{op.player_index}-{thread_no}"

        with open(mpspdz_input_file, "a") as f:
            for arg in inputs.keys():
                f.write(str(inputs[arg]) + " ")
        get_logger().debug(
            f"Executing MpspdzSaveInputKernel, "
            f"op:{op}, "
            f"session_id:{session_id}, "
            f"inputs:{inputs}"
        )
        return 0


class MpspdzCallKernel(Kernel):
    async def execute(self, op, session_id, output, **kwargs):
        concrete_kwargs = {key: await value for key, value in kwargs.items()}
        control_inputs = concrete_kwargs

        get_logger().debug(
            f"Executing MpspdzCallKernel, session_id:{session_id}, inputs:{control_inputs}"
        )
        assert isinstance(op, MpspdzCallOperation)
        prog_name = await self.write_bytecode(await self.compile_to_mpc(op.mlir))

        cmd2 = f"./mpspdz-links.sh {session_id} {op.invocation_key}"
        proc2 = await asyncio.create_subprocess_shell(
            cmd2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc2.communicate()
 
        mpspdz_executable = "./mascot-party.x"
        args = [
            mpspdz_executable,
            "-p",
            str(op.player_index),
            "-N",
            "3",
            "-h",
            "inputter0",
            "-pn",
            "12000",
            os.path.splitext(prog_name)[0],
        ]
        args = f"./mascot-party.x -p {op.player_index} -N 3 -h inputter0"

        get_logger().debug(f"Running external program: {args}")

        p = pathlib.Path("/MP-SPDZ")

        cmd = "cd /MP-SPDZ;" + args
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        get_logger().debug(f"Running external program: Done")

        output.set_result(0)

    async def compile_to_mpc(self, mlir, elk_binary="./elk-to-mpc"):
        with tempfile.NamedTemporaryFile(mode="wt") as mlir_file:
            with tempfile.NamedTemporaryFile(mode="rt", delete=False) as mpc_file:

                mlir_file.write(mlir)
                mlir_file.flush()

                args = [
                    elk_binary,
                    mlir_file.name,
                    "-o",
                    mpc_file.name,
                ]
                get_logger().debug(f"Running external program: {args}")
                _ = subprocess.run(
                    args, stdout=subprocess.PIPE, universal_newlines=True,
                )

                mpc_with_main = mpc_file.read() + "\n" + "main()"
                return mpc_with_main

    async def write_bytecode(self, mpc, mpspdz_compiler="./compile.py"):
        mpc_file_name = None
        with tempfile.NamedTemporaryFile(
            mode="wt", suffix=".mpc", dir="/MP-SPDZ/Programs/Source", delete=False
        ) as mpc_file:
            mpc_file.write(mpc)
            mpc_file.flush()

            args = [
                mpspdz_compiler,
                mpc_file.name,
            ]
            mpc_file_name = mpc_file.name.split("/")[-1]
            get_logger().debug(f"Running external program: {args}")
            _ = subprocess.run(args, stdout=subprocess.PIPE, universal_newlines=True, cwd='/MP-SPDZ')
        return mpc_file_name


class MpspdzLoadOutputKernel(Kernel):
    def execute_synchronous_block(self, op, session_id, **control_inputs):
        assert isinstance(op, MpspdzLoadOutputOperation)
        get_logger().debug(
            f"Executing MpspdzLoadOutputKernel, session_id:{session_id}, inputs:{control_inputs}"
        )
        # this is a bit ugly, inspiration from here: https://github.com/data61/MP-SPDZ/issues/104
        # but really, it can be much nicer if the flag in
        # https://github.com/data61/MP-SPDZ/blob/master/Processor/Instruction.hpp#L1229
        # is set to true (ie human readable)

        # in the future we might need to use the print_ln_to instruction

        # default value of prime
        prime = 170141183460469231731687303715885907969
        # default R of Montgomery representation
        R = 2 ** 128
        # Inverse mod prime to get clear value Integer
        invR = 96651956244403355751989957128965938997

        mpspdz_dir = (
            f"/MP-SPDZ/tmp/{session_id}/{op.invocation_key}/Player-Data/Private-Output"
        )
        mpspdz_output_file = f"{mpspdz_dir}-{op.player_index}"

        get_logger().debug(f"Reading Private-Output")
        outputs = list()
        with open(mpspdz_output_file, "rb") as f:
            while (byte := f.read(16)) :
                # As integer
                tmp = int.from_bytes(byte, byteorder="little")
                # Invert "Montgomery"
                clear = (tmp * invR) % prime
                outputs.append(clear)

        get_logger().debug(
            f"Executing LoadOutputCallKernel, op:{op}, session_id:{session_id}, inputs:{control_inputs}"
        )
        get_logger().debug("XXXXXXXXXXXXXXXXXXX OUTPUTS")
        get_logger().debug(f"results are: {outputs}")
        # TODO return actual value
        return outputs


class KernelBasedExecutor:
    def __init__(self, name, channel_manager, store={}):
        self.name = name
        self.store = store
        self.kernels = {
            LoadOperation: LoadKernel(store),
            SaveOperation: SaveKernel(store),
            SendOperation: SendKernel(channel_manager),
            ReceiveOperation: ReceiveKernel(channel_manager),
            DeserializeOperation: DeserializeKernel(),
            SerializeOperation: SerializeKernel(),
            ConstantOperation: ConstantKernel(),
            AddOperation: AddKernel(),
            SubOperation: SubKernel(),
            MulOperation: MulKernel(),
            DivOperation: DivKernel(),
            RunProgramOperation: RunProgramKernel(),
            CallPythonFunctionOperation: CallPythonFunctionKernel(),
            MpspdzSaveInputOperation: MpspdzSaveInputKernel(),
            MpspdzCallOperation: MpspdzCallKernel(),
            MpspdzLoadOutputOperation: MpspdzLoadOutputKernel(),
        }

    def compile_computation(self, logical_computation):
        # TODO for now we don't do any compilation of computations
        return logical_computation

    async def run_computation(self, logical_computation, placement, session_id):
        physical_computation = self.compile_computation(logical_computation)
        execution_plan = self.schedule_execution(physical_computation, placement)
        # link futures together using kernels
        session_values = AsyncStore()
        tasks = []
        for op in execution_plan:
            kernel = self.kernels.get(type(op))
            if not kernel:
                raise NotImplementedError(f"No kernel found for operation {type(op)}")

            inputs = {
                param_name: session_values.get_future(key=value_name)
                for (param_name, value_name) in op.inputs.items()
            }
            output = session_values.get_future(key=op.output) if op.output else None
            tasks += [
                asyncio.create_task(
                    kernel.execute(op, session_id=session_id, output=output, **inputs)
                )
            ]
        # execute kernels
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        # address any errors that may have occurred
        exceptions = [task.exception() for task in done if task.exception()]
        for e in exceptions:
            get_logger().exception(e)
        if exceptions:
            raise Exception(f"One or more errors occurred in '{self.name}'")

    def schedule_execution(self, comp, placement):
        # TODO(Morten) this is as simple and naive as it gets; we should at least
        # do some kind of topology sorting to make sure we have all async values
        # ready for linking with kernels in `run_computation`
        return [node for node in comp.nodes() if node.device_name == placement]


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
