import argparse
import logging

from moose import edsl
from moose.choreography.grpc import Choreographer
from moose.logger import get_logger

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = edsl.host_placement(name="inputter0")
inputter1 = edsl.host_placement(name="inputter1")
aggregator = edsl.host_placement(name="aggregator")
outputter = edsl.host_placement(name="outputter")


@edsl.function
def mul_fn(x, y):
    return x * y


@edsl.computation
def my_comp():

    with inputter0:
        c0_0 = edsl.constant(1)
        c1_0 = edsl.constant(2)
        x0 = mul_fn(c0_0, c1_0)

    with inputter1:
        c0_1 = edsl.constant(2)
        c1_1 = edsl.constant(3)
        x1 = mul_fn(c0_1, c1_1)

    with aggregator:
        y = edsl.add(x0, x1)

    with outputter:
        res = edsl.save("y", y)

    return res


concrete_comp = edsl.trace(my_comp)

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
