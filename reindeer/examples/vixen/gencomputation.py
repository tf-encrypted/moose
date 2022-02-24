#!/usr/bin/env python3


import numpy as np
import pandas as pd

from pymoose import edsl, elk_compiler
from pymoose.computation import utils

player0 = edsl.host_placement("player0")
player1 = edsl.host_placement("player1")
player2 = edsl.host_placement("player2")
repl = edsl.replicated_placement("replicated", [player0, player1, player2])


@edsl.computation
def linear_predict():
    with player0:
        #df = pd.read_csv("data.csv")
        df = pd.read_csv("big_data.csv")
        data = df.to_numpy()[:256]
        biased = np.vstack([np.ones(len(data)), data.T]).T
        x = edsl.constant(biased)
        x = edsl.cast(x, dtype=edsl.fixed(14, 23))
    with player2:
        w = edsl.constant(
            np.array([1, 2, 3, 4], dtype=np.float64).reshape((4, 1))
        )
        w = edsl.cast(w, dtype=edsl.fixed(14, 23))
    with repl:
        y_hat = edsl.dot(x, w)
        #for _ in range(400):
        #    y_hat = edsl.mul(y_hat, y_hat)
    with player1:
        result = edsl.cast(y_hat, dtype=edsl.float64)
    return result


def comp_to_disk(filename):
    concrete_comp = edsl.trace(linear_predict)
    comp_bin = utils.serialize_computation(concrete_comp)
    rust_compiled = elk_compiler.compile_computation(comp_bin)
    with open(filename, "wb") as f:
        f.write(rust_compiled.to_bytes())


if __name__ == "__main__":
    comp_to_disk("comp.bytes")
