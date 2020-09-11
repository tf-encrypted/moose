import logging

from compiler.edsl import HostPlacement
from compiler.edsl import add
from compiler.edsl import computation
from compiler.edsl import constant
from compiler.edsl import run_python_script
from compiler.edsl import save
from compiler.logger import get_logger
from compiler.runtime import TestRuntime

get_logger().setLevel(level=logging.DEBUG)

inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
aggregator = HostPlacement(name="aggregator")
outputter = HostPlacement(name="outputter")


@computation
def my_comp():

    with inputter0:
        c0_0 = constant(2)
        c1_0 = constant(3)
        x0 = run_python_script("local_computation.py", c0_0, c1_0)

    with inputter1:
        c0_1 = constant(3)
        x1 = run_python_script("local_computation.py", c0_1)

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
        placement_assignment={
            inputter0: runtime.executors[0],
            inputter1: runtime.executors[1],
            aggregator: runtime.executors[2],
            outputter: runtime.executors[3],
        },
    )

    print("Done")
