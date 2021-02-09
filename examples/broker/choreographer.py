import argparse
import logging

import moose as moo
from moose.choreography.grpc import Choreographer
from moose.logger import get_logger

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = moo.host_placement(name="inputter0")
inputter1 = moo.host_placement(name="inputter1")
aggregator = moo.host_placement(name="aggregator")
outputter = moo.host_placement(name="outputter")


@moo.function
def mul_fn(x, y):
    return x * y


@moo.computation
def my_comp():

    with inputter0:
        c0_0 = moo.constant(1)
        c1_0 = moo.constant(2)
        x0 = mul_fn(c0_0, c1_0)

    with inputter1:
        c0_1 = moo.constant(2)
        c1_1 = moo.constant(3)
        x1 = mul_fn(c0_1, c1_1)

    with aggregator:
        y = moo.add(x0, x1)

    with outputter:
        res = moo.save("y", y)

    return res


concrete_comp = moo.trace(my_comp)

if __name__ == "__main__":
    runtime = Choreographer()
    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_instantiation={
            inputter0: "worker0:50000",
            inputter1: "worker1:50000",
            aggregator: "worker2:50000",
            outputter: "worker3:50000",
        },
    )

    print("Done")
