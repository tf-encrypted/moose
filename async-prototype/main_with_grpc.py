from grpc.experimental import aio
import logging

from logger import get_logger
from logger import set_logger


from channels.grpc_channels import Channel
from channels.grpc_channels import ChannelManager
from edsl import Role
from edsl import add
from edsl import constant
from edsl import computation
from edsl import load
from edsl import mul
from edsl import save
from edsl import sub
from executor import AsyncKernelBasedExecutor
from runtime import RemoteRuntime

from grpc.experimental import aio


get_logger().setLevel(level=logging.DEBUG)

inputter0 = Role(name="inputter0")
inputter1 = Role(name="inputter1")
aggregator = Role(name="aggregator")
outputter = Role(name="outputter")

cluster_spec = {
    inputter0.name: "localhost:50051",
    inputter1.name: "localhost:50052",
    aggregator.name: "localhost:50053",
    outputter.name: "localhost:50054",
}


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


concrete_comp = my_comp.trace_func()
print(concrete_comp)

runtime = RemoteRuntime(cluster_spec)
runtime.evaluate_computation(concrete_comp)
print("Done")
