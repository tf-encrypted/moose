import argparse
import logging
import os

from moose.choreography.grpc import Choreographer as GrpcChoreographer
from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import function
from moose.compiler.edsl import save
from moose.logger import get_logger
from moose.runtime import TestRuntime
from moose.utils import load_certificate

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--ca-cert", default=os.environ.get("CA_CERT", None))
parser.add_argument("--ident-cert", default=os.environ.get("IDENT_CERT", None))
parser.add_argument("--ident-key", default=os.environ.get("IDENT_KEY", None))
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
aggregator = HostPlacement(name="aggregator")
outputter = HostPlacement(name="outputter")


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
        res = save(y, "y")

    return res


concrete_comp = my_comp.trace_func()

if __name__ == "__main__":
    if args.runtime == "test":
        runtime = TestRuntime()
    elif args.runtime == "remote":
        runtime = GrpcChoreographer(
            ca_cert=load_certificate(args.ca_cert),
            ident_cert=load_certificate(args.ident_cert),
            ident_key=load_certificate(args.ident_key),
        )
    else:
        raise ValueError(f"Unknown runtime '{args.runtime}'")

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
