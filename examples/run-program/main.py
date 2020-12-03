import logging

from moose.edsl import add
from moose.edsl import computation
from moose.edsl import constant
from moose.edsl import host_placement
from moose.edsl import run_program
from moose.edsl import save
from moose.edsl import trace
from moose.logger import get_logger
from moose.runtime import TestRuntime

get_logger().setLevel(level=logging.DEBUG)

inputter0 = host_placement(name="inputter0")
inputter1 = host_placement(name="inputter1")
aggregator = host_placement(name="aggregator")
outputter = host_placement(name="outputter")


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
