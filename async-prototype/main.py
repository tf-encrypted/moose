from channels import AsyncChannelManager
from computation import Computation
from edsl import Role
from edsl import add
from edsl import computation
from edsl import constant
from edsl import load
from edsl import mul
from edsl import save
from edsl import sub
from executor import AsyncKernelBasedExecutor
from runtime import Runtime
from logger import get_logger
import logging

get_logger().setLevel(level=logging.DEBUG)

inputter0 = Role(name="inputter0")
inputter1 = Role(name="inputter1")
aggregator = Role(name="aggregator")
outputter = Role(name="outputter")


@computation
def my_comp():

    with inputter0:
        x0 = constant(5)

    with inputter1:
        x1 = constant(7)

    with aggregator:
        y0 = add(x0, x0)
        y1 = mul(x1, x1)
        y = add(y0, y1)

    with outputter:
        res = save(y, "y")

    return res


channel_manager = AsyncChannelManager()

in0_executor = AsyncKernelBasedExecutor(name="alice", channel_manager=channel_manager)
in1_executor = AsyncKernelBasedExecutor(name="bob", channel_manager=channel_manager)
agg_executor = AsyncKernelBasedExecutor(name="carole", channel_manager=channel_manager)
out_executor = AsyncKernelBasedExecutor(name="dave", channel_manager=channel_manager)

runtime = Runtime(
    role_assignment={
        inputter0: in0_executor,
        inputter1: in1_executor,
        aggregator: agg_executor,
        outputter: out_executor,
    }
)

concrete_comp = my_comp.trace_func()
runtime.evaluate_computation(concrete_comp)
print("Done")
