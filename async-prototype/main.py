import argparse
import logging

from edsl import Role
from edsl import add
from edsl import computation
from edsl import constant
from edsl import mul
from edsl import save
from logger import get_logger
from runtime import RemoteRuntime
from runtime import TestRuntime

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
parser.add_argument(
    "--cluster-spec", default="cluster/cluster-spec-docker-compose.yaml"
)
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = Role(name="inputter0")
inputter1 = Role(name="inputter1")
aggregator = Role(name="aggregator")
outputter = Role(name="outputter")


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
        runtime = RemoteRuntime("cluster/cluster-spec-localhost.yaml")
        assert len(runtime.executors) == len(concrete_comp.devices())
    else:
        raise ValueError(f"Unknown runtime '{args.runtime}'")

    runtime.evaluate_computation(
        computation=concrete_comp,
        role_assignment={
            inputter0: runtime.executors[0],
            inputter1: runtime.executors[1],
            aggregator: runtime.executors[2],
            outputter: runtime.executors[3],
        },
    )

    print("Done")
