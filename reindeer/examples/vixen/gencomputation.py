#!/usr/bin/env python3


import numpy as np
import pandas as pd

import pymoose as pm
from pymoose.computation import utils

player0 = pm.host_placement("player0")
player1 = pm.host_placement("player1")
player2 = pm.host_placement("player2")
repl = pm.replicated_placement("replicated", [player0, player1, player2])


@pm.computation
def linear_predict():
    with player0:
        df = pd.read_csv("data.csv")
        data = df.to_numpy()
        biased = np.vstack([np.ones(len(data)), data.T]).T
        x = pm.constant(biased)
        x = pm.cast(x, dtype=pm.fixed(14, 23))
    with player2:
        w = pm.constant(
            np.array([1, 2, 3, 4], dtype=np.float64).reshape((4, 1))
        )
        w = pm.cast(w, dtype=pm.fixed(14, 23))
    with repl:
        y_hat = pm.dot(x, w)
    with player1:
        result = pm.cast(y_hat, dtype=pm.float64)
    return result


def comp_to_disk(filename):
    concrete_comp = pm.trace(linear_predict)
    comp_bin = utils.serialize_computation(concrete_comp)
    rust_compiled = pm.elk_compiler.compile_computation(comp_bin)
    with open(filename, "wb") as f:
        f.write(rust_compiled.to_bytes())


if __name__ == "__main__":
    comp_to_disk("comp.bytes")
