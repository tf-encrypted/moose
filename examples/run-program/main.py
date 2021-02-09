import logging

import moose as moo
from moose.logger import get_logger
from moose.testing import TestRuntime

get_logger().setLevel(level=logging.DEBUG)

inputter0 = moo.host_placement(name="inputter0")
inputter1 = moo.host_placement(name="inputter1")
aggregator = moo.host_placement(name="aggregator")
outputter = moo.host_placement(name="outputter")


@moo.computation
def my_comp():

    with inputter0:
        c0_0 = moo.constant(2)
        c1_0 = moo.constant(3)
        x0 = moo.run_program("python", ["local_computation.py"], c0_0, c1_0)

    with inputter1:
        c0_1 = moo.constant(3)
        x1 = moo.run_program("python", ["local_computation.py"], c0_1)

    with aggregator:
        y = moo.add(x0, x1)

    with outputter:
        res = moo.save("y", y)

    return res


concrete_comp = moo.trace(my_comp)

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
