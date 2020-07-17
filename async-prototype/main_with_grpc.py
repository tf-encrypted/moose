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


inputter0 = Role(name="inputter0")
aggregator = Role(name="aggregator")
outputter = Role(name="outputter")


@computation
def my_comp():

    with inputter0:
        x0 = load("x0")
        x = add(x0, x0)

    with aggregator:
        y = add(x, x)

    with outputter:
        res = save(y, "y0")

    return res


concrete_comp = my_comp.trace_func()
print(concrete_comp)

# Currently, it's using the same Channel client to add the value to the grpc server
# buffer (when inputter send), and for the outputter to receive.
# Probably need two different Channel clients but with same endpoint.
channel_inp_agg = Channel("localhost", "50051")
channel_agg_out = Channel("localhost", "50052")

channels = {("inputter0", "aggregator"): channel_inp_agg,
            ("aggregator", "outputter"): channel_agg_out}

channel_manager = ChannelManager(channels)

in0_executor = AsyncKernelBasedExecutor(
    name="alice", store={"x0": 5}, channel_manager=channel_manager,
)
agg_executor = AsyncKernelBasedExecutor(
    name="Mike", store={}, channel_manager=channel_manager
)
out_executor = AsyncKernelBasedExecutor(
    name="dave", store={}, channel_manager=channel_manager
)

runtime = Runtime(role_assignment={inputter0: in0_executor, 
                                   aggregator: agg_executor,
                                   outputter: out_executor,})

runtime.evaluate_computation(concrete_comp)
print("Done")
