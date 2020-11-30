import asyncio
import dataclasses
from typing import Optional

from moose.computation.host import CallPythonFunctionOperation
from moose.computation.host import RunProgramOperation
from moose.computation.mpspdz import MpspdzCallOperation
from moose.computation.mpspdz import MpspdzLoadOutputOperation
from moose.computation.mpspdz import MpspdzSaveInputOperation
from moose.computation.standard import AddOperation
from moose.computation.standard import ConstantOperation
from moose.computation.standard import DeserializeOperation
from moose.computation.standard import DivOperation
from moose.computation.standard import LoadOperation
from moose.computation.standard import MulOperation
from moose.computation.standard import ReceiveOperation
from moose.computation.standard import SaveOperation
from moose.computation.standard import SendOperation
from moose.computation.standard import SerializeOperation
from moose.computation.standard import SubOperation
from moose.executor.kernels.host import CallPythonFunctionKernel
from moose.executor.kernels.host import RunProgramKernel
from moose.executor.kernels.mpspdz import MpspdzCallKernel
from moose.executor.kernels.mpspdz import MpspdzLoadOutputKernel
from moose.executor.kernels.mpspdz import MpspdzSaveInputKernel
from moose.executor.kernels.standard import AddKernel
from moose.executor.kernels.standard import ConstantKernel
from moose.executor.kernels.standard import DeserializeKernel
from moose.executor.kernels.standard import DivKernel
from moose.executor.kernels.standard import LoadKernel
from moose.executor.kernels.standard import MulKernel
from moose.executor.kernels.standard import ReceiveKernel
from moose.executor.kernels.standard import SaveKernel
from moose.executor.kernels.standard import SendKernel
from moose.executor.kernels.standard import SerializeKernel
from moose.executor.kernels.standard import SubKernel
from moose.logger import get_logger
from moose.storage import AsyncStore


@dataclasses.dataclass
class Session:
    session_id: int
    placement_instantiation: Optional
    values: AsyncStore = dataclasses.field(repr=False)


class AsyncExecutor:
    def __init__(self, networking, store={}):
        self.store = store
        self.kernels = {
            ConstantOperation: ConstantKernel(),
            AddOperation: AddKernel(),
            SubOperation: SubKernel(),
            MulOperation: MulKernel(),
            DivOperation: DivKernel(),
            LoadOperation: LoadKernel(store),
            SaveOperation: SaveKernel(store),
            SendOperation: SendKernel(networking),
            ReceiveOperation: ReceiveKernel(networking),
            SerializeOperation: SerializeKernel(),
            DeserializeOperation: DeserializeKernel(),
            RunProgramOperation: RunProgramKernel(),
            CallPythonFunctionOperation: CallPythonFunctionKernel(),
            MpspdzSaveInputOperation: MpspdzSaveInputKernel(),
            MpspdzCallOperation: MpspdzCallKernel(networking),
            MpspdzLoadOutputOperation: MpspdzLoadOutputKernel(),
        }

    def compile_computation(self, logical_computation):
        # TODO for now we don't do any compilation of computations
        return logical_computation

    async def run_computation(
        self, logical_computation, placement_instantiation, placement, session_id
    ):
        physical_computation = self.compile_computation(logical_computation)
        execution_plan = self.schedule_execution(physical_computation, placement)
        session = Session(
            session_id=session_id,
            placement_instantiation=placement_instantiation,
            values=AsyncStore(),
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
