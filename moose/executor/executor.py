import asyncio
import dataclasses
from typing import Dict

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
from moose.executor.kernels.mpspdz import MpspdzCallKernel
from moose.executor.kernels.mpspdz import MpspdzLoadOutputKernel
from moose.executor.kernels.mpspdz import MpspdzSaveInputKernel
from moose.executor.kernels.standard import AddKernel
from moose.executor.kernels.standard import CallPythonFunctionKernel
from moose.executor.kernels.standard import ConstantKernel
from moose.executor.kernels.standard import DeserializeKernel
from moose.executor.kernels.standard import DivKernel
from moose.executor.kernels.standard import LoadKernel
from moose.executor.kernels.standard import MulKernel
from moose.executor.kernels.standard import ReceiveKernel
from moose.executor.kernels.standard import RunProgramKernel
from moose.executor.kernels.standard import SaveKernel
from moose.executor.kernels.standard import SendKernel
from moose.executor.kernels.standard import SerializeKernel
from moose.executor.kernels.standard import SubKernel
from moose.logger import get_logger
from moose.storage import AsyncStore


@dataclasses.dataclass
class Session:
    session_id: int
    placement_instantiation: Dict[str, str]
    values: AsyncStore = dataclasses.field(repr=False)


class AsyncExecutor:
    def __init__(self, networking, store={}):
        self.store = store
        self.kernels = {
            LoadOperation: LoadKernel(store),
            SaveOperation: SaveKernel(store),
            SendOperation: SendKernel(networking),
            ReceiveOperation: ReceiveKernel(networking),
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
            output = session.values.get_future(key=op.output) if op.output else None
            tasks += [
                asyncio.create_task(
                    kernel.execute(op, session=session, output=output, **inputs)
                )
            ]
        get_logger().debug(f"Exiting computation; session_id:{session.session_id}")
        # check that there's something to do since `asyncio.wait` will block otherwise
        if not tasks:
            get_logger().warn(f"Computation had no tasks; session_id:{session.session_id}")
            return
        # execute kernels
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        # address any errors that may have occurred
        exceptions = [task.exception() for task in done if task.exception()]
        for e in exceptions:
            get_logger().exception(e)
        if exceptions:
            raise Exception(f"One or more errors occurred in '{placement}'")

    def schedule_execution(self, comp, placement):
        # TODO(Morten) this is as simple and naive as it gets; we should at least
        # do some kind of topology sorting to make sure we have all async values
        # ready for linking with kernels in `run_computation`
        return [node for node in comp.nodes() if node.device_name == placement]
