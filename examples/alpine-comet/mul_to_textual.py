import pathlib

import numpy as np

import pymoose as pm
from pymoose.computation import utils as comp_utils

parent_dir = pathlib.Path(__file__).parent
save_file = parent_dir / "repl_mul.moose"

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
nitrogen = pm.host_placement("nitrogen")
replicated = pm.replicated_placement("replicated", [alice, bob, nitrogen])

@pm.computation
def main():
    with alice:
        x = pm.constant(np.array([1., 2., 3., 4.]), dtype=pm.fixed(24, 40))
    with bob:
        y = pm.constant(np.array([5., 6., 7., 8.]), dtype=pm.fixed(24, 40))

    with replicated:
        z = x * y + x
    
    with bob:
        res = pm.cast(z, dtype=pm.float64)

    return res


if __name__ == "__main__":
    comp = pm.trace(main)
    comp_ser = comp_utils.serialize_computation(comp)
    comp_ser_rs = pm.elk_compiler.compile_computation(comp_ser)
    comp_textual = comp_ser_rs.to_textual()

    with open(save_file, "w") as f:
        f.write(comp_textual)
