import argparse
import logging

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import mul
from moose.compiler.edsl import save
from moose.logger import get_logger
from moose.runtime import RemoteRuntime
from moose.runtime import TestRuntime

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
parser.add_argument(
    "--cluster-spec", default="./moose/cluster/cluster-spec-localhost.yaml"
)
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
aggregator = HostPlacement(name="aggregator")
outputter = HostPlacement(name="outputter")


@computation
def my_comp():

    with inputter0:
        x0 = constant(5)

    with inputter1:
        x1 = constant(7)

    with aggregator:
        y0 = add(x0, x0)
        y1 = mul(x1, x1)
        y = add(y0, y1)

    with outputter:
        res = save(y, "y")

    return res


concrete_comp = my_comp.trace_func()


if __name__ == "__main__":

    if args.runtime == "test":
        runtime = TestRuntime(num_workers=len(concrete_comp.devices()))
    elif args.runtime == "remote":
        runtime = RemoteRuntime(args.cluster_spec)
        assert len(runtime.executors) == len(concrete_comp.devices())
    else:
        raise ValueError(f"Unknown runtime '{args.runtime}'")

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_assignment={
            inputter0: runtime.executors[0],
            inputter1: runtime.executors[1],
            aggregator: runtime.executors[2],
            outputter: runtime.executors[3],
        },
    )

    print("Done")
