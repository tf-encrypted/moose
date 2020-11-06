import asyncio

from moose.logger import get_logger


class Kernel:
    async def execute(self, op, session, output, **kwargs):
        concrete_kwargs = {key: await value for key, value in kwargs.items()}
        get_logger().debug(
            f"Executing:"
            f" kernel:{self.__class__.__name__},"
            f" op:{op},"
            f" session:{session},"
            f" inputs:{concrete_kwargs}"
        )
        concrete_output = self.execute_synchronous_block(
            op=op, session=session, **concrete_kwargs
        )
        get_logger().debug(
            f"Done executing:"
            f" kernel:{self.__class__.__name__},"
            f" op:{op},"
            f" session:{session},"
            f" output:{concrete_output}"
        )
        if output:
            output.set_result(concrete_output)

    def execute_synchronous_block(self, op, session, **kwargs):
        raise NotImplementedError()


async def run_external_program(args, cwd=None):
    get_logger().debug(f"Run external program, launching: args:{args}, cwd:{cwd}")
    cmd = " ".join(args)
    proc = await asyncio.create_subprocess_shell(
        cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    get_logger().debug(f"[{cmd!r} exited with {proc.returncode}]")
    if stdout:
        get_logger().debug(f"[stdout]\n{stdout.decode()}")
    if stderr:
        get_logger().debug(f"[stderr]\n{stderr.decode()}")
    get_logger().debug("Run external program, finished")
