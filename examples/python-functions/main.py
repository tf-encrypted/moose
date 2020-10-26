import argparse
import logging
import os

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import function
from moose.compiler.edsl import save
from moose.logger import get_logger
from moose.runtime import RemoteRuntime
from moose.runtime import TestRuntime

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--cluster-spec", default="cluster-spec.yaml")
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
        runtime = TestRuntime(workers=concrete_comp.devices())
    elif args.runtime == "remote":
        runtime = RemoteRuntime(
            args.cluster_spec,
            ca_cert_filename=args.ca_cert,
            ident_cert_filename=args.ident_cert,
            ident_key_filename=args.ident_key,
        )
        assert set(concrete_comp.devices()).issubset(runtime.executors.keys())
    else:
        raise ValueError(f"Unknown runtime '{args.runtime}'")

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_assignment={
            inputter0: runtime.executors["inputter0"],
            inputter1: runtime.executors["inputter1"],
            aggregator: runtime.executors["aggregator"],
            outputter: runtime.executors["outputter"],
        },
    )

    print("Done")
