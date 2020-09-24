import argparse
import logging

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import call
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
    "--cluster-spec", default="cluster/cluster-spec-docker-compose.yaml"
)
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
aggregator = HostPlacement(name="aggregator")
outputter = HostPlacement(name="outputter")


@computation
def inner_comp(x0, x1):
    five = constant(5, placement=inputter0)
    x0_prime = add(x0, five, placement=inputter0)
    y = add(x0_prime, x1, placement=aggregator)
    return y

@computation
def outer_comp():
    x0 = constant(5, placement=inputter0)
    x1 = constant(7, placement=inputter1)
    y = call(inner_comp, args=(x0, x1), placements=[inputter0, aggregator])
    res = save(y, "y", placement=outputter)
    return res


concrete_comp = outer_comp.trace_func()


if __name__ == "__main__":

    if args.runtime == "test":
        runtime = TestRuntime(num_workers=len(concrete_comp.devices()))
    elif args.runtime == "remote":
        runtime = RemoteRuntime("moose/cluster/cluster-spec-localhost.yaml")
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
