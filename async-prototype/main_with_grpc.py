from channels.grpc_channels import Channel
from channels.grpc_channels import ChannelManager
from edsl import Role
from edsl import add
from edsl import computation
from edsl import load
from edsl import mul
from edsl import save
from edsl import sub
from executor import AsyncKernelBasedExecutor
from runtime import Runtime

from grpc.experimental import aio


inputter0 = Role(name="inputter0")
inputter1 = Role(name="inputter1")
aggregator = Role(name="aggregator")
outputter = Role(name="outputter")


@computation
def my_comp():

    with inputter0:
        x0 = load("x0")

    with inputter1:
        x1 = load("x1")

    with aggregator:
        y0 = add(x0, x0)
        y1 = mul(x1, x1)
        y = add(y0, y1)

    with outputter:
        res = save(y, "y")

    return res


concrete_comp = my_comp.trace_func()
print(concrete_comp)

cluster_spec = {
    inputter0.name: "localhost:50051",
    inputter1.name: "localhost:50052",
    aggregator.name: "localhost:50053",
}

channel_manager = ChannelManager(cluster_spec)

in0_executor = AsyncKernelBasedExecutor(
    name="alice", store={"x0": 5}, channel_manager=channel_manager,
)
in1_executor = AsyncKernelBasedExecutor(
    name="bob", store={"x1": 7}, channel_manager=channel_manager,
)
agg_executor = AsyncKernelBasedExecutor(
    name="carole", store={}, channel_manager=channel_manager,
)
out_executor = AsyncKernelBasedExecutor(
    name="dave", store={}, channel_manager=channel_manager
)


runtime = Runtime(
    role_assignment={
        inputter0: in0_executor,
        inputter1: in1_executor,
        aggregator: agg_executor,
        outputter: out_executor,
    }
)

runtime.evaluate_computation(concrete_comp)
print("Done")
