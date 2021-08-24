import argparse
import logging

from pymoose import edsl
from pymoose.deprecated import edsl as old_edsl
from pymoose.deprecated.choreography.grpc import Choreographer as GrpcChoreographer
from pymoose.deprecated.testing import TestRuntime
from pymoose.logger import get_logger

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

inputter0 = edsl.host_placement(name="inputter0")
inputter1 = edsl.host_placement(name="inputter1")
outputter = edsl.host_placement(name="outputter")

# NOTE:
# All players must be listed in the MP-SPDZ placement, even if they only send
# inputs or receive outputs (and don't perform compute). This is because the
# setup for the placement needs to know ahead of time who to generate key pairs
# for. In the near future this is ideally something that we can infer automati-
# cally during compilation from logical to physical computation.
mpspdz = old_edsl.mpspdz_placement(
    name="mpspdz", players=[inputter0, inputter1, outputter]
)


@old_edsl.function
def my_function(x, y, z):
    return x * y + z


@edsl.computation
def my_comp():

    with inputter0:
        x = edsl.constant(1)
        z = edsl.constant(3)

    with inputter1:
        y = edsl.constant(2)

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

    with outputter:
        res = edsl.save("v", edsl.add(v, w))

    return res


concrete_comp = edsl.tracer.trace_and_compile(my_comp)

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
        },
    )

    print("Done")
