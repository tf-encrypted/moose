import asyncio
import binascii
import hashlib
import os
import tempfile

from moose.compiler.computation import MpspdzCallOperation
from moose.compiler.computation import MpspdzLoadOutputOperation
from moose.compiler.computation import MpspdzSaveInputOperation
from moose.executor.base import Kernel
from moose.executor.base import run_external_program
from moose.logger import get_logger


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


class MpspdzCallKernel(Kernel):
    async def execute(self, op, session_id, output, **control_inputs):
        assert isinstance(op, MpspdzCallOperation)
        _ = await asyncio.gather(*control_inputs.values())

        await prepare_mpspdz_directory(op, session_id)

        with tempfile.NamedTemporaryFile(
            mode="wt", suffix=".mpc", dir="/MP-SPDZ/Programs/Source", delete=False,
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
        # Inverse mod prime to get clear value Integer
        invR = 96651956244403355751989957128965938997  # (2^128^-1 mod prime)

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
