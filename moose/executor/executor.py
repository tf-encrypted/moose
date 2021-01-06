import asyncio
import dataclasses
from typing import Any

from moose.computation import fixedpoint as fixed_ops
from moose.computation import host as host_ops
from moose.computation import mpspdz as mpspdz_ops
from moose.computation import primitives as primitives_ops
from moose.computation import ring as ring_ops
from moose.computation import standard as standard_ops
from moose.executor.kernels import fixedpoint as fixed_kernels
from moose.executor.kernels import host as host_kernels
from moose.executor.kernels import mpspdz as mpspdz_kernels
from moose.executor.kernels import primitives as primitives_kernels
from moose.executor.kernels import ring as ring_kernels
from moose.executor.kernels import standard as standard_kernels
from moose.logger import get_logger
from moose.storage import AsyncStore


@dataclasses.dataclass
class Session:
    session_id: int
    placement_instantiation: Any
    values: AsyncStore = dataclasses.field(repr=False)
    arguments: AsyncStore = dataclasses.field(repr=False)


class AsyncExecutor:
    def __init__(self, networking, store={}):
        self.store = store
        self.kernels = {
            standard_ops.InputOperation: standard_kernels.InputKernel(),
            standard_ops.OutputOperation: standard_kernels.OutputKernel(),
            standard_ops.ConstantOperation: standard_kernels.ConstantKernel(),
            standard_ops.AddOperation: standard_kernels.AddKernel(),
            standard_ops.SubOperation: standard_kernels.SubKernel(),
            standard_ops.MulOperation: standard_kernels.MulKernel(),
            standard_ops.DivOperation: standard_kernels.DivKernel(),
            standard_ops.TransposeOperation: standard_kernels.TransposeKernel(),
            ring_ops.FillTensorOperation: ring_kernels.RingFillKernel(),
            ring_ops.RingAddOperation: ring_kernels.RingAddKernel(),
            ring_ops.RingDotOperation: ring_kernels.RingDotKernel(),
            ring_ops.RingMulOperation: ring_kernels.RingMulKernel(),
            ring_ops.RingSampleOperation: ring_kernels.RingSampleKernel(),
            ring_ops.RingSubOperation: ring_kernels.RingSubKernel(),
            ring_ops.RingShapeOperation: ring_kernels.RingShapeKernel(),
            primitives_ops.DeriveSeedOperation: primitives_kernels.DeriveSeedKernel(),
            primitives_ops.SampleKeyOperation: primitives_kernels.SampleKeyKernel(),
            standard_ops.LoadOperation: standard_kernels.LoadKernel(store),
            standard_ops.SaveOperation: standard_kernels.SaveKernel(store),
            standard_ops.SendOperation: standard_kernels.SendKernel(networking),
            standard_ops.ReceiveOperation: standard_kernels.ReceiveKernel(networking),
            standard_ops.SerializeOperation: standard_kernels.SerializeKernel(),
            standard_ops.DeserializeOperation: standard_kernels.DeserializeKernel(),
            host_ops.RunProgramOperation: host_kernels.RunProgramKernel(),
            host_ops.CallPythonFunctionOperation: (
                host_kernels.CallPythonFunctionKernel()
            ),
            mpspdz_ops.MpspdzSaveInputOperation: mpspdz_kernels.MpspdzSaveInputKernel(),
            mpspdz_ops.MpspdzCallOperation: mpspdz_kernels.MpspdzCallKernel(networking),
            mpspdz_ops.MpspdzLoadOutputOperation: (
                mpspdz_kernels.MpspdzLoadOutputKernel()
            ),
            fixed_ops.RingEncodeOperation: (fixed_kernels.RingEncodeKernel()),
            fixed_ops.RingDecodeOperation: (fixed_kernels.RingDecodeKernel()),
        }

    def compile_computation(self, logical_computation):
        # TODO for now we don't do any compilation of computations
        return logical_computation

    async def run_computation(
        self,
        logical_computation,
        placement_instantiation,
        placement,
        session_id,
        arguments={},
    ):
        physical_computation = self.compile_computation(logical_computation)
        execution_plan = self.schedule_execution(physical_computation, placement)
        session = Session(
            session_id=session_id,
            placement_instantiation=placement_instantiation,
            values=AsyncStore(),
            arguments=AsyncStore(initial_values=arguments),
        )
        get_logger().debug(
            f"Entering computation; placement:{placement}, session:{session}"
        )
        # link futures together using kernels
        tasks = []
        for op in execution_plan:
            kernel = self.kernels.get(type(op))
            if not kernel:
                raise NotImplementedError(f"No kernel found for operation {type(op)}")

            inputs = {
                param_name: session.values.get_future(key=value_name)
                for (param_name, value_name) in op.inputs.items()
            }
            output = session.values.get_future(key=op.name)
            tasks += [
                asyncio.create_task(
                    kernel.execute(op, session=session, output=output, **inputs)
                )
            ]
        get_logger().debug(f"Exiting computation; session_id:{session.session_id}")
        # check that there's something to do since `asyncio.wait` will block otherwise
        if not tasks:
            get_logger().warn(
                f"Computation had no tasks; session_id:{session.session_id}"
            )
            return
        # execute kernels
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        # address any errors that may have occurred
        exceptions = [task.exception() for task in done if task.exception()]
        for e in exceptions:
            get_logger().exception(e)
        if exceptions:
            raise Exception(
                f"One or more errors occurred in '{placement}: {exceptions}'"
            )

    def schedule_execution(self, comp, placement):
        # TODO(Morten) this is as simple and naive as it gets; we should at least
        # do some kind of topology sorting to make sure we have all async values
        # ready for linking with kernels in `run_computation`
        return [op for op in comp.operations.values() if op.placement_name == placement]
