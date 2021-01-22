import argparse
import logging

from moose.choreography.grpc import Choreographer
from moose.edsl import add
from moose.edsl import computation
from moose.edsl import constant
from moose.edsl import function
from moose.edsl import host_placement
from moose.edsl import save
from moose.edsl import trace
from moose.logger import get_logger

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = host_placement(name="inputter0")
inputter1 = host_placement(name="inputter1")
aggregator = host_placement(name="aggregator")
outputter = host_placement(name="outputter")


@function
def mul_fn(x, y):
    return x * y


@computation
def my_comp():

    with inputter0:
        c0_0 = constant(1)
        c1_0 = constant(2)
        x0 = mul_fn(c0_0, c1_0)

    with inputter1:
        c0_1 = constant(2)
        c1_1 = constant(3)
        x1 = mul_fn(c0_1, c1_1)

    with aggregator:
        y = add(x0, x1)

    with outputter:
        res = save("y", y)

    return res


concrete_comp = trace(my_comp)

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
