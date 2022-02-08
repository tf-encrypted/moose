#!/usr/bin/env python3


import numpy as np
from pymoose import edsl, elk_compiler
from pymoose.computation import dtypes, utils

player0 = edsl.host_placement("player0")
player1 = edsl.host_placement("player1")
player2 = edsl.host_placement("player2")
repl = edsl.replicated_placement("replicated", [player0, player1, player2])


def bias_trick(X):
    """Appends column of 1s to input matrix X."""
    bias_shape = edsl.slice(edsl.shape(X), begin=0, end=1)
    bias = edsl.ones(bias_shape, dtype=edsl.float64)
    reshaped_bias = edsl.expand_dims(bias, 1)
    X_b = edsl.concatenate([reshaped_bias, X], axis=1)
    return X_b


@edsl.computation
def linear_predict(
    # TODO [Kyle]: Will need to set the type of the x input data to a new
    # AES tensor type that Morten is working on. Will possibly have even
    # more input parameters
    x: edsl.Argument(placement=player0, dtype=dtypes.float64)
):
    with player0:
        x = bias_trick(x)
        x = edsl.cast(x, dtype=edsl.fixed(14, 23))
    with player2:
        w = edsl.constant(
            np.array(
                [1, 2, 3, 4],
                dtype=np.float64,
            ).reshape((4, 1))
        )
        w = edsl.cast(w, dtype=edsl.fixed(14, 23))
    with repl:
        y_hat = edsl.dot(x, w)
    with player1:
        result = edsl.cast(y_hat, dtype=edsl.float64)
    return result


def comp_to_disk(filename):
    concrete_comp = edsl.trace(linear_predict)
    comp_bin = utils.serialize_computation(concrete_comp)
    rust_compiled = elk_compiler.compile_computation(
        comp_bin,
        [
            "typing",
            "full",
            "prune",
            "networking",
            "typing",
        ],
    )
    with open(filename, "wb") as f:
        f.write(rust_compiled.to_bytes())


if __name__ == "__main__":
    comp_to_disk("comp.bytes")
