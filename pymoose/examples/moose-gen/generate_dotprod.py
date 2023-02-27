import pathlib

import numpy as np

import pymoose as pm
from pymoose.computation import utils

FIXED = pm.fixed(24, 40)
player0 = pm.host_placement("player0")
player1 = pm.host_placement("player1")
player2 = pm.host_placement("player2")
repl = pm.replicated_placement("replicated", [player0, player1, player2])

this_dir = pathlib.Path(__file__).parent


@pm.computation
def my_computation():
    with player0:
        x = pm.constant(np.array([1.0, 2.0, 3.0]).reshape((1, 3)))
        x = pm.cast(x, dtype=FIXED)
    with player1:
        w = pm.constant(np.array([4.0, 5.0, 6.0]).reshape((3, 1)))
        w = pm.cast(w, dtype=FIXED)

    with repl:
        y_hat = pm.dot(x, w)

    with player2:
        result = pm.cast(y_hat, dtype=pm.float64)

    return result


def comp_to_moose(filepath):
    traced_comp: pm.edsl.base.AbstractComputation = pm.trace(my_computation)
    comp_bin: bytes = utils.serialize_computation(traced_comp)
    rust_comp: pm.MooseComputation = pm.elk_compiler.compile_computation(
        comp_bin, passes=[]
    )
    textual_comp: str = rust_comp.to_textual()
    with open(filepath, "w") as f:
        f.write(textual_comp)


if __name__ == "__main__":
    comp_to_moose(this_dir / "dotprod.moose")
