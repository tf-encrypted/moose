import asyncio
from dataclasses import dataclass
from typing import Dict, List

from computation import comp
from executor import AsyncKernelBasedExecutor
from executor import AsyncMemoryChannel


in0_agg = AsyncMemoryChannel()
in1_agg = AsyncMemoryChannel()
agg_out = AsyncMemoryChannel()

channels = {
    "in0_agg": in0_agg,
}

in0_executor = AsyncKernelBasedExecutor(
    name="inputter0", store={"x0": 5}, channels={"in0_agg": in0_agg,}
)
in1_executor = AsyncKernelBasedExecutor(
    name="inputter1", store={"x1": 7}, channels={"in1_agg": in1_agg,}
)
agg_executor = AsyncKernelBasedExecutor(
    name="aggregator",
    store={},
    channels={"in0_agg": in0_agg, "in1_agg": in1_agg, "agg_out": agg_out,},
)
out_executor = AsyncKernelBasedExecutor(
    name="outputter", store={}, channels={"agg_out": agg_out,}
)

in0_executor.run_computation(comp, in0_device.name, "123")

# role_assignment = {
#     in0_device: in0_executor,
#     in1_device: in1_executor,
#     agg_device: agg_executor,
#     out_device: out_executor,
# }


# channels = [
#     (in0_device, agg_device):
# ]
