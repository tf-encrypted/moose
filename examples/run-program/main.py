import logging

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import run_program
from moose.compiler.edsl import save
from moose.logger import get_logger
from moose.runtime import TestRuntime

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
        x0 = run_program("python", ["local_computation.py"], c0_0, c1_0)

    with inputter1:
        c0_1 = constant(3)
        x1 = run_program("python", ["local_computation.py"], c0_1)

    with aggregator:
        y = add(x0, x1)

    with outputter:
        res = save(y, "y")

    return res


concrete_comp = my_comp.trace_func()

if __name__ == "__main__":

    runtime = TestRuntime()

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_instantiation={
            inputter0.name: "inputter0",
            inputter1.name: "inputter1",
            aggregator.name: "aggregator",
            outputter.name: "outputter",
        },
    )

    print("Done")
