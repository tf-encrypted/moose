import asyncio
import binascii
import hashlib
import tempfile
from pathlib import Path

from moose.compiler.computation import MpspdzCallOperation
from moose.compiler.computation import MpspdzLoadOutputOperation
from moose.compiler.computation import MpspdzSaveInputOperation
from moose.executor.base import Kernel
from moose.executor.base import run_external_program
from moose.logger import get_logger


def prepare_mpspdz_directory(
    op, session_id, mpspdz_dirname="/MP-SPDZ", protocol_name=None
):
    mpspdz = Path(mpspdz_dirname)
    root = (
        Path(tempfile.gettempdir())
        / str(op.device_name)
        / str(session_id)
        / str(op.invocation_key)
    )

    if not root.exists():
        root.mkdir(parents=True)

    player_data = root / "Player-Data"
    if not player_data.exists():
        player_data.mkdir()

    script_filename = None
    if protocol_name is not None:
        script_filename = f"{op.protocol}-party.x" 
        script = root / script_filename
        if not script.exists():
            # TODO can this be a symlink instead?
            # all mpspdz executables are statically built
            (mpspdz / f'static' / script_filename).link_to(script)

    programs = root / "Programs"
    if not programs.exists():
        programs.symlink_to(mpspdz / "Programs")

    return root, mpspdz, script_filename


class MpspdzSaveInputKernel(Kernel):
    async def execute(self, op, session_id, output, **inputs):
        assert isinstance(op, MpspdzSaveInputOperation)
        concrete_inputs = {key: await value for key, value in inputs.items()}

        isolated_dir, _, _ = prepare_mpspdz_directory(op=op, session_id=session_id)

        thread_no = 0  # assume inputs are happening in the main thread
        input_filename = str(
            isolated_dir / "Player-Data" / f"Input-P{op.player_index}-{thread_no}"
        )
        get_logger().debug(f"Saving inputs to {input_filename}")

        with open(input_filename, "a") as f:
            # TODO make sure we sort by key
            for key, value in concrete_inputs.items():
                f.write(str(value) + " ")

        output.set_result(0)


class MpspdzCallKernel(Kernel):
    def __init__(self, channel_manager):
        self.channel_manager = channel_manager

    async def execute(self, op, session_id, output, **control_inputs):
        assert isinstance(op, MpspdzCallOperation)
        _ = await asyncio.gather(*control_inputs.values())

        isolated_dir, mpspdz_dir, script_filename = prepare_mpspdz_directory(
            op=op, session_id=session_id, protocol_name=op.protocol
        )

        mlir_filename = str(isolated_dir / "source.mlir")
        with open(mlir_filename, "wt") as mlir_file:
            mlir_file.write(op.mlir)
        get_logger().debug(f"Wrote .mlir file: {mlir_filename}")

        mpc_filename = str(isolated_dir / "source.mpc")
        await run_external_program(
            args=["./elk-to-mpc", mlir_filename, "-o", mpc_filename]
        )
        with open(mpc_filename, "at") as mpc_file:
            mpc_file.write("\n" + "main()")
        get_logger().debug(f"Wrote .mpc file: {mpc_filename}")

        program_name = f"{session_id}-{op.invocation_key}-{op.player_index}"
        mpc_symlink = mpspdz_dir / "Programs" / "Source" / f"{program_name}.mpc"
        mpc_symlink.symlink_to(mpc_filename)
        get_logger().debug(f"Linked {mpc_symlink.name} to {mpc_filename}")

        await run_external_program(
            args=["./compile.py", mpc_symlink.name], cwd=str(mpspdz_dir)
        )
        get_logger().debug(f"Compiled program: {program_name}")

        # TODO: replace hostname with localhost if it's testruntime
        await run_external_program(
            cwd=str(isolated_dir),
            args=[
                f"./{script_filename}",
                "--player",
                str(op.player_index),
                "--nparties",
                str(op.num_players),
                "--hostname",
                self.channel_manager.get_hostname(op.coordinator),
                "--portnumbase",
                str(self.derive_port_number(op, session_id)),
                program_name,
            ],
        )

        output.set_result(0)

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

        isolated_dir, _, _ = prepare_mpspdz_directory(op=op, session_id=session_id)

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

        output_filename = (
            isolated_dir / "Player-Data" / f"Private-Output-{op.player_index}"
        )
        get_logger().debug(f"Loading values from {output_filename}")

        outputs = list()
        with open(output_filename, "rb") as output_file:
            while True:
                byte = output_file.read(16)
                if not byte:
                    break
                # As integer
                tmp = int.from_bytes(byte, byteorder="little")
                # Invert "Montgomery"
                clear = (tmp * invR) % prime
                outputs.append(clear)

        return outputs
