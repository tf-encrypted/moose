import pathlib

import numpy as np

import pymoose as pm

FIXED = pm.fixed(14, 23)

alice = pm.host_placement(name="alice")
bob = pm.host_placement(name="bob")
carole = pm.host_placement(name="carole")
rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])


@pm.computation
def psi_and_agg():
    with alice:
        x_a = pm.load("x_a", dtype=pm.float64)
        x_a = pm.cast(x_a, dtype=FIXED)
        in_this_db_a = pm.load("in_this_db_a", dtype=pm.bool_)

    with bob:
        x_b = pm.load("x_b", dtype=pm.float64)
        in_this_db_b = pm.load("in_this_db_b", dtype=pm.bool_)

        # Check if the key is in both Alice and Bob's datasets
        # TODO implemented logical_and on host
        intersect_bool = pm.logical_and(in_this_db_a, in_this_db_b)

        # Extract data subset from Bob's data based on intersected keys
        x_b_sub = pm.index_axis(x_b, axis=0, index=intersect_bool)
        x_b_sub = pm.cast(x_b_sub, dtype=FIXED)

    with alice:
        # Extract data subsets from Alice's data based on intersected keys
        x_a_sub = pm.index_axis(x_b, axis=0, index=intersect_bool)
        x_a_sub = pm.cast(x_a_sub, dtype=FIXED)

    with rep:
        # Aggregation: average ratio between x_a_sub & x_b_sub
        res  = pm.div(pm.sum(x_a_sub, axis=0), pm.sum(x_b_sub, axis=0))

    with alice:
        res = pm.cast(res, dtype=pm.float64)
        res = pm.save("agg_result", res)

    return res


if __name__ == "__main__":
    _DATA_DIR = pathlib.Path(__file__).parent / "data"

    x_a = np.load(_DATA_DIR / "x_a.npy")
    x_b = np.load(_DATA_DIR / "x_b.npy")
    in_this_db_a = np.load(_DATA_DIR / "in_this_db_a.npy")
    in_this_db_b = np.load(_DATA_DIR / "in_this_db_b.npy")

    executors_storage = {
        "alice": {"x_a": x_a, "in_this_db_a": in_this_db_a},
        "bob": {"x_b": x_b, "in_this_db_b": in_this_db_b},
    }

    runtime = pm.LocalMooseRuntime(
        identities=["alice", "bob", " carole"],
        storage_mapping=executors_storage,
    )

    runtime.set_default()

    _ = psi_and_agg()

    agg_result = runtime.read_value_from_storage("alice", "agg_result")
    print("Aggregation Result", agg_result)