import argparse
import logging

from moose.choreography.grpc import Choreographer as GrpcChoreographer
from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import function
from moose.compiler.edsl import save
from moose.compiler.mpspdz import MpspdzPlacement
from moose.logger import get_logger
from moose.runtime import TestRuntime

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
outputter = HostPlacement(name="outputter")
saver = HostPlacement(name="saver")

# NOTE:
# All players must be listed in the MP-SPDZ placement, even if they only send
# inputs or receive outputs (and don't perform compute). This is because the
# setup for the placement needs to know ahead of time who to generate key pairs
# for. In the near future this is ideally something that we can infer automati-
# cally during compilation from logical to physical computation.
mpspdz = MpspdzPlacement(name="mpspdz", players=[inputter0, inputter1, outputter])


@function
def my_function(x, y, z):
    return x * y + z


@computation
def my_comp():

    with inputter0:
        x = constant(1)
        z = constant(3)

    with inputter1:
        y = constant(2)

    with mpspdz:
        # Note that this illustrates one issue with function calls:
        # what does the role assignment indicate? is it where the
        # function is evaluated (in which case, how to we specify
        # placement of (revealed) outputs)? or is it the placement
        # of outputs (in which case, how do we deal with multiple
        # outputs on different placements)? here we are opting for
        # the former which seems to match better with graphs.
        #
        # Note also that we want to infer full type signatures in
        # the future, which should include expected output type and
        # hence placement information, making this less of an issue.
        #
        # Finally, note that the two function applications are being
        # executed concurrently in different sessions by MP-SPDZ.
        v = my_function(x, y, z, output_placements=[outputter])
        w = my_function(x, y, z, output_placements=[outputter])

    with saver:
        res = save(add(v, w), "v")

    return res


concrete_comp = my_comp.trace_func()

if __name__ == "__main__":
    if args.runtime == "test":
        runtime = TestRuntime()
    elif args.runtime == "remote":
        runtime = GrpcChoreographer()
    else:
        raise ValueError(f"Unknown runtime '{args.runtime}'")

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_instantiation={
            inputter0: "inputter0:50000",
            inputter1: "inputter1:50000",
            outputter: "outputter:50000",
            saver: "saver:50000",
        },
    )

    print("Done")
