import logging

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import MpspdzPlacement
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import function
from moose.compiler.edsl import save
from moose.logger import get_logger
from moose.runtime import TestRuntime

get_logger().setLevel(level=logging.DEBUG)

inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
outputter = HostPlacement(name="outputter")
mpspdz = MpspdzPlacement(name="mpspdz", players=[inputter0, inputter1])


@function
def my_function(x, y):
    return x * y


@computation
def my_comp():

    with inputter0:
        x = constant(1)

    with inputter1:
        y = constant(2)

    with mpspdz:
        # note that this illustrates one issue with function calls:
        # what does the role assignment indicate? is it where the
        # function is evaluated (in which case, how to we specify
        # placement of (revealed) outputs)? or is it the placement
        # of outputs (in which case, how do we deal with multiple
        # outputs on different placements)? here we are opting for
        # the former which seems to match better with graphs.
        #
        # note also that we want to infer full type signatures in
        # the future, which should include expected output type and
        # hence placement information, making this less of an issue.
        z = my_function(x, y, output_placements=[outputter])

    with outputter:
        res = save(z, "z")

    return res


concrete_comp = my_comp.trace_func()

if __name__ == "__main__":

    runtime = TestRuntime(num_workers=len(concrete_comp.devices()))

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_assignment={
            inputter0: runtime.executors[0],
            inputter1: runtime.executors[1],
            outputter: runtime.executors[2],
        },
    )

    print("Done")
