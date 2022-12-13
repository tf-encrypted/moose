import pathlib

import numpy as np

import pymoose as pm

FIXED = pm.fixed(24, 40)

alice = pm.host_placement(name="alice")
bob = pm.host_placement(name="bob")
carole = pm.host_placement(name="carole")
rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])


@pm.computation
def psi_and_agg():
    with alice:
        x_a = pm.load("x_a", dtype=pm.float64)
        user_id_available_a = pm.load("user_id_available_a", dtype=pm.bool_)

    with bob:
        x_b = pm.load("x_b", dtype=pm.float64)
        user_id_available_b = pm.load("user_id_available_b", dtype=pm.bool_)

        # # Compute logical And between user_id_available from Alice and Bob.
        # # If it returns 1, it means the User ID was in Alice and Bob's datasets
        exist_in_alice_and_bob_bool = pm.logical_and(
            user_id_available_a, user_id_available_b
        )

        # # Filter Bob's feature to keep only records where exist_in_alice_and_bob_bool returned 1
        x_b_sub = pm.select(x_b, axis=0, index=exist_in_alice_and_bob_bool)
        x_b_sub = pm.cast(x_b_sub, dtype=FIXED)

    with alice:
        # Filter Alice's feature to keep only records where exist_in_alice_and_bob_bool returned 1
        x_a_sub = pm.select(x_a, axis=0, index=exist_in_alice_and_bob_bool)
        x_a_sub = pm.cast(x_a_sub, dtype=FIXED)

    with rep:
        # Aggregation: average ratio between sum of x_a_sub & x_b_sub
        res = pm.div(pm.sum(x_a_sub, axis=0), pm.sum(x_b_sub, axis=0))

    with alice:
        res = pm.cast(res, dtype=pm.float64)
        res = pm.save("agg_result", res)

    return res


if __name__ == "__main__":
    _DATA_DIR = pathlib.Path(__file__).parent / "data"

    x_a = np.load(_DATA_DIR / "x_a.npy")
    x_b = np.load(_DATA_DIR / "x_b.npy")

    user_id_available_a = np.load(_DATA_DIR / "user_id_available_a.npy").astype(
        np.bool_
    )
    user_id_available_b = np.load(_DATA_DIR / "user_id_available_b.npy").astype(
        np.bool_
    )

    executors_storage = {
        "alice": {"x_a": x_a, "user_id_available_a": user_id_available_a},
        "bob": {"x_b": x_b, "user_id_available_b": user_id_available_b},
    }

    runtime = pm.LocalMooseRuntime(
        identities=["alice", "bob", "carole"],
        storage_mapping=executors_storage,
    )

    runtime.set_default()

    _ = psi_and_agg()

    agg_result = runtime.read_value_from_storage("alice", "agg_result")
    print("Aggregation Result", agg_result)
