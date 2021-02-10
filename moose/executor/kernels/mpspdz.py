import asyncio
import binascii
import hashlib
import tempfile
from pathlib import Path

from moose.computations.mpspdz import MpspdzCallOperation
from moose.computations.mpspdz import MpspdzLoadOutputOperation
from moose.computations.mpspdz import MpspdzSaveInputOperation
from moose.executor.kernels.base import Kernel
from moose.executor.kernels.base import run_external_program
from moose.logger import get_logger
from moose.logger import get_tracer


def prepare_mpspdz_directory(
    op, session_id, mpspdz_dirname="/MP-SPDZ", protocol_name=None
):
    mpspdz = Path(mpspdz_dirname)
    root = (
        Path(tempfile.gettempdir())
        / str(op.placement_name)
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
            (mpspdz / "static" / script_filename).link_to(script)

    programs = root / "Programs"
    if not programs.exists():
        programs.symlink_to(mpspdz / "Programs")

    return root, mpspdz, script_filename


class MpspdzSaveInputKernel(Kernel):
    async def execute(self, op, session, output, **inputs):
        assert isinstance(op, MpspdzSaveInputOperation)
        concrete_inputs = {key: await value for key, value in inputs.items()}

        with get_tracer().start_as_current_span(f"{op.name}"):
            isolated_dir, _, _ = prepare_mpspdz_directory(
                op=op, session_id=session.session_id
            )

            thread_no = 0  # assume inputs are happening in the main thread
            input_filename = str(
                isolated_dir / "Player-Data" / f"Input-P{op.player_index}-{thread_no}"
            )
            get_logger().debug(f"Saving inputs to {input_filename}")

            with open(input_filename, "a") as f:
                for key in sorted(concrete_inputs.keys()):
                    value = concrete_inputs[key]
                    f.write(str(value) + " ")

            output.set_result(0)


class MpspdzCallKernel(Kernel):
    def __init__(self, networking):
        self.networking = networking

    async def execute(self, op, session, output, **control_inputs):
        assert isinstance(op, MpspdzCallOperation)
        _ = await asyncio.gather(*control_inputs.values())

        with get_tracer().start_as_current_span(f"{op.name}"):
            isolated_dir, mpspdz_dir, script_filename = prepare_mpspdz_directory(
                op=op, session_id=session.session_id, protocol_name=op.protocol
            )

            mlir_filename = str(isolated_dir / "source.mlir")
            with open(mlir_filename, "wt") as mlir_file:
                mlir_file.write(op.mlir)
            get_logger().debug(f"Wrote .mlir file: {mlir_filename}")

            mpc_filename = str(isolated_dir / "source.mpc")
            await run_external_program(
                args=["elk-to-mpc", mlir_filename, "-o", mpc_filename]
            )
            with open(mpc_filename, "at") as mpc_file:
                mpc_file.write("\n" + "main()")
            get_logger().debug(f"Wrote .mpc file: {mpc_filename}")

            program_name = f"{session.session_id}-{op.invocation_key}-{op.player_index}"
            mpc_symlink = mpspdz_dir / "Programs" / "Source" / f"{program_name}.mpc"
            mpc_symlink.symlink_to(mpc_filename)
            get_logger().debug(f"Linked {mpc_symlink.name} to {mpc_filename}")

            await run_external_program(
                args=["./compile.py", mpc_symlink.name], cwd=str(mpspdz_dir)
            )
            get_logger().debug(f"Compiled program: {program_name}")

            await run_external_program(
                cwd=str(isolated_dir),
                args=[
                    f"./{script_filename}",
                    "--player",
                    str(op.player_index),
                    "--nparties",
                    str(op.num_players),
                    "--hostname",
                    self.networking.get_hostname(
                        session.placement_instantiation.get(op.coordinator)
                    ),
                    "--portnumbase",
                    str(self.derive_port_number(op, session.session_id)),
                    "--output-file",
                    "Player-Data/Private-Output",
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
    def execute_synchronous_block(self, op, session, **control_inputs):
        assert isinstance(op, MpspdzLoadOutputOperation)

        with get_tracer().start_as_current_span(f"{op.name}"):
            isolated_dir, _, _ = prepare_mpspdz_directory(
                op=op, session_id=session.session_id
            )

            output_filename = (
                isolated_dir / "Player-Data" / f"Private-Output-P{op.player_index}-0"
            )
            get_logger().debug(f"Loading values from {output_filename}")

            outputs = list()
            with open(output_filename, "r") as output_file:
                for line in output_file.readlines():
                    # For now everything is an integer
                    outputs.append(int(line))

            return outputs
