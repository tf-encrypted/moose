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
    "--cluster-spec", default="cluster/cluster-spec-docker-compose.yaml"
)
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = Principal(name="inputter0")
inputter1 = Principal(name="inputter1")

inputter0_plc = HostPlacement(principal=inputter0)
inputter1_plc = HostPlacement(principal=inputter1)
aggregator_plc = HostPlacement(name="aggregator")
outputter_plc = HostPlacement(name="outputter")
mpspdz_plc = MpspdzPlacement([aggregator_plc])


@function
def foo(x, y):
    return x + y

@computation
def my_inner_comp(y):
    z = mul(y, 2, placement=y.placement)
    return z

@computation
def my_outer_comp():
    x0 = constant(5, placement=inputter0_plc)
    x1 = constant(7, placement=inputter1_plc)
    y = apply(foo, (x0, x1), placement=mpspdz_plc)  # may compile into call
    z = call(my_inner_comp, y)


concrete_comp = my_comp.trace_func()


if __name__ == "__main__":

    if args.runtime == "test":
        runtime = TestRuntime(num_workers=len(concrete_comp.devices()))
    elif args.runtime == "remote":
        runtime = RemoteRuntime("cluster/cluster-spec-localhost.yaml")
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
