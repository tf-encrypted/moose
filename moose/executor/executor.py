import asyncio
import binascii
import hashlib
import json
import os
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
        get_logger().debug(
            f"Executing:"
            f" kernel:{self.__class__.__name__},"
            f" op:{op},"
            f" session_id:{session_id},"
            f" inputs:{concrete_kwargs}"
        )
        concrete_output = self.execute_synchronous_block(
            op=op, session_id=session_id, **concrete_kwargs
        )
        get_logger().debug(
            f"Done executing:"
            f" kernel:{self.__class__.__name__},"
            f" op:{op},"
            f" session_id:{session_id},"
            f" output:{concrete_output}"
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

        args = ["./mpspdz-links.sh", f"{session_id}", f"{op.invocation_key}"]
        subprocess.call(args)

        thread_no = 0  # assume inputs are happening in the main thread
        mpspdz_input_file = (
            f"/MP-SPDZ/tmp/{session_id}/{op.invocation_key}/"
            f"Player-Data/Input-P{op.player_index}-{thread_no}"
        )

        with open(mpspdz_input_file, "a") as f:
            for arg in inputs.keys():
                f.write(str(inputs[arg]) + " ")

        return 0


async def prepare_mpspdz_directory(op, session_id):
    await run_external_program(
        args=["./mpspdz-links.sh", str(session_id), str(op.invocation_key)]
    )


async def run_external_program(args, cwd=None):
    get_logger().debug(f"Run external program, launch: {args}")
    cmd = " ".join(args)
    proc = await asyncio.create_subprocess_shell(
        cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    get_logger().debug(f"Run external program, finish: {args}")


class MpspdzCallKernel(Kernel):
    async def execute(self, op, session_id, output, **control_inputs):
        assert isinstance(op, MpspdzCallOperation)
        _ = await asyncio.gather(*control_inputs.values())

        await prepare_mpspdz_directory(op, session_id)

        with tempfile.NamedTemporaryFile(
            mode="wt", suffix=".mpc", dir="/MP-SPDZ/Programs/Source"
        ) as mpc_file:
            mpc = await self.compile_mpc(op.mlir)
            mpc_file.write(mpc)
            mpc_file.flush()
            bytecode_filename = await self.compile_and_write_bytecode(mpc_file.name)
            await self.run_mpspdz(
                player_index=op.player_index,
                port_number=self.derive_port_number(op, session_id),
                bytecode_filename=bytecode_filename,
            )

        output.set_result(0)

    async def compile_mpc(self, mlir):
        with tempfile.NamedTemporaryFile(mode="rt") as mpc_file:
            with tempfile.NamedTemporaryFile(mode="wt") as mlir_file:
                mlir_file.write(mlir)
                mlir_file.flush()
                await run_external_program(
                    args=["./elk-to-mpc", mlir_file.name, "-o", mpc_file.name]
                )
                mpc_without_main = mpc_file.read()
                return mpc_without_main + "\n" + "main()"

    async def compile_and_write_bytecode(self, mpc_filename):
        await run_external_program(
            args=["./compile.py", mpc_filename], cwd="/MP-SPDZ",
        )
        return os.path.splitext(mpc_filename.split("/")[-1])[0]

    async def run_mpspdz(self, player_index, port_number, bytecode_filename):
        await run_external_program(
            args=[
                "cd /MP-SPDZ; ./mascot-party.x",
                "-p",
                str(player_index),
                "-N",
                "3",
                "-h",
                "inputter0",
                "-pn",
                str(port_number),
                bytecode_filename,
            ]
        )

    def derive_port_number(self, op, session_id, min_port=10000, max_port=20000):
        h = hashlib.new("sha256")
        h.update(f"{session_id} {op.invocation_key}".encode("utf-8"))
        hash_value = binascii.hexlify(h.digest())
        # map into [min_port; max_port)
        port_number = int(hash_value, 16) % (max_port - min_port) + min_port
        return port_number


class MpspdzLoadOutputKernel(Kernel):
    def execute_synchronous_block(self, op, session_id, **control_inputs):
        assert isinstance(op, MpspdzLoadOutputOperation)

        # this is a bit ugly, inspiration from here:
        # https://github.com/data61/MP-SPDZ/issues/104
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
        get_logger().debug(f"Loading values from {mpspdz_output_file}")

        outputs = list()
        with open(mpspdz_output_file, "rb") as f:
            while True:
                byte = f.read(16)
                if not byte:
                    break
                # As integer
                tmp = int.from_bytes(byte, byteorder="little")
                # Invert "Montgomery"
                clear = (tmp * invR) % prime
                outputs.append(clear)
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
