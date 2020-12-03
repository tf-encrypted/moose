import logging

from moose.computation import HostPlacement
from moose.edsl import add
from moose.edsl import computation
from moose.edsl import constant
from moose.edsl import default_placement
from moose.edsl import run_program
from moose.edsl import save
from moose.edsl import trace
from moose.logger import get_logger
from moose.runtime import TestRuntime

get_logger().setLevel(level=logging.DEBUG)

inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
aggregator = HostPlacement(name="aggregator")
outputter = HostPlacement(name="outputter")


@computation
def my_comp():

    with default_placement(inputter0):
        c0_0 = constant(2)
        c1_0 = constant(3)
        x0 = run_program("python", ["local_computation.py"], c0_0, c1_0)

    with default_placement(inputter1):
        c0_1 = constant(3)
        x1 = run_program("python", ["local_computation.py"], c0_1)

    with default_placement(aggregator):
        y = add(x0, x1)

    with default_placement(outputter):
        res = save(y, "y")

    return res


concrete_comp = trace(my_comp)

if __name__ == "__main__":

    runtime = TestRuntime()

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_instantiation={
            inputter0: "inputter0",
            inputter1: "inputter1",
            aggregator: "aggregator",
            outputter: "outputter",
        },
    )

    print("Done")
