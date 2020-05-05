import asyncio
from dataclasses import dataclass
from typing import Dict
from typing import List

from edsl import Role
from edsl import add
from edsl import computation
from edsl import load
from edsl import save
from executor import AsyncKernelBasedExecutor
from executor import AsyncMemoryChannel


in0_agg = AsyncMemoryChannel()
in1_agg = AsyncMemoryChannel()
agg_out = AsyncMemoryChannel()

in0_executor = AsyncKernelBasedExecutor(
    name="alice", store={"x0": 5}, channels={"inputter0_aggregator": in0_agg}, send_delay=2,
)
in1_executor = AsyncKernelBasedExecutor(
    name="bob", store={"x1": 7}, channels={"inputter1_aggregator": in1_agg}, send_delay=None,
)
agg_executor = AsyncKernelBasedExecutor(
    name="carole",
    store={},
    channels={
        "inputter0_aggregator": in0_agg,
        "inputter1_aggregator": in1_agg,
        "aggregator_outputter": agg_out,
    },
)
out_executor = AsyncKernelBasedExecutor(
    name="dave", store={}, channels={"aggregator_outputter": agg_out}
)


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
        y1 = add(x1, x1)
        y = add(y0, y1)

    with outputter:
        res = save(y, "y")

    # TODO(Morten) remove; we only need this to define root expression
    return res


_ = my_comp(
    role_assignment={
        inputter0: in0_executor,
        inputter1: in1_executor,
        aggregator: agg_executor,
        outputter: out_executor,
    }
)
