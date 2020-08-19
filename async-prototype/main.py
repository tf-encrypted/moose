import logging

from edsl import Role
from edsl import add
from edsl import computation
from edsl import constant
from edsl import mul
from edsl import save
from logger import get_logger
from runtime import TestRuntime

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


concrete_comp = my_comp.trace_func()

runtime = TestRuntime(num_workers=4)
runtime.evaluate_computation(
    computation=concrete_comp,
    role_assignment={
        inputter0: runtime.executors[0],
        inputter1: runtime.executors[1],
        aggregator: runtime.executors[2],
        outputter: runtime.executors[3],
    },
)

print("Done")
