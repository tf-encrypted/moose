import logging

from moose import edsl
from moose.deprecated import edsl as old_edsl
from moose.deprecated.testing import TestRuntime
from moose.logger import get_logger

get_logger().setLevel(level=logging.DEBUG)

inputter0 = edsl.host_placement(name="inputter0")
inputter1 = edsl.host_placement(name="inputter1")
aggregator = edsl.host_placement(name="aggregator")
outputter = edsl.host_placement(name="outputter")


@edsl.computation
def my_comp():

    with inputter0:
        c0_0 = edsl.constant(2)
        c1_0 = edsl.constant(3)
        x0 = old_edsl.run_program("python", ["local_computation.py"], c0_0, c1_0)

    with inputter1:
        c0_1 = edsl.constant(3)
        x1 = old_edsl.run_program("python", ["local_computation.py"], c0_1)

    with aggregator:
        y = edsl.add(x0, x1)

    with outputter:
        res = edsl.save("y", y)

    return res


concrete_comp = edsl.tracer.trace_and_compile(my_comp)

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
