import logging

from computation import Computation
from edsl import Role
from edsl import add
from edsl import call_program
from edsl import computation
from edsl import constant
from edsl import load
from edsl import mul
from edsl import save
from edsl import sub
from logger import get_logger
from logger import set_logger
from runtime import RemoteRuntime
from runtime import TestRuntime

get_logger().setLevel(level=logging.DEBUG)

inputter0 = Role(name="inputter0")
inputter1 = Role(name="inputter1")
aggregator = Role(name="aggregator")
outputter = Role(name="outputter")


@computation
def my_comp():

    with inputter0:
        c0 = constant(2)
        x0 = call_program("local_computation.py", c0)

    with inputter1:
        c1 = constant(3)
        x1 = call_program("local_computation.py", c1)

    with aggregator:
        y = add(x0, x1)

    with outputter:
        res = save(y, "y")

    return res


concrete_comp = my_comp.trace_func()

if __name__ == "__main__":

    runtime = TestRuntime(num_workers=len(concrete_comp.devices()))

    runtime.evaluate_computation(
        computation=concrete_comp,
        role_assignment={
            inputter0: runtime.executors[0],
            inputter1: runtime.executors[1],
            aggregator: runtime.executors[2],
            outputter: runtime.executors[3],
            outputter: runtime.executors[1],
        },
    )

    print("Done")
