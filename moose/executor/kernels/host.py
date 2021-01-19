import asyncio
import json
import subprocess
import tempfile

import dill

from moose.computation.host import CallPythonFunctionOperation
from moose.computation.host import RunProgramOperation
from moose.executor.kernels.base import Kernel
from moose.logger import get_logger
from moose.logger import get_tracer


class CallPythonFunctionKernel(Kernel):
    async def execute(self, op, session, output, **inputs):
        assert isinstance(op, CallPythonFunctionOperation)
        concrete_inputs = await asyncio.gather(*inputs.values())
        with get_tracer().start_as_current_span(f"{op.name}"):
            python_fn = dill.loads(op.pickled_fn)
            concrete_output = python_fn(*concrete_inputs)
            output.set_result(concrete_output)


class RunProgramKernel(Kernel):
    async def execute(self, op, session, output, **inputs):
        assert isinstance(op, RunProgramOperation)
        concrete_inputs = await asyncio.gather(*inputs.values())
        with get_tracer().start_as_current_span(f"{op.name}"):

            with tempfile.NamedTemporaryFile() as inputfile:
                with tempfile.NamedTemporaryFile() as outputfile:

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
