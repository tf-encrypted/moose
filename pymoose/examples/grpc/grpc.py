import argparse
import logging

import numpy as np

import pymoose as pm
from pymoose.logger import get_logger

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
carole = pm.host_placement("carole")


@pm.computation
def my_computation(
    v: pm.Argument(placement=alice, vtype=pm.TensorType(pm.float64)),
):
    with alice:
        x = pm.constant(np.array([1.0, 2.0], dtype=np.float64))

    with bob:
        y = pm.constant(np.array([3.0, 4.0], dtype=np.float64))

    with carole:
        z = pm.add(x, y)
        w = pm.add(z, v)

    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    identity_map = {
        alice.name: "localhost:50000",
        bob.name: "localhost:50001",
        carole.name: "localhost:50002",
    }

    runtime = pm.GrpcMooseRuntime(identity_map)
    runtime.set_default()

    results = my_computation(np.array([5.0, 6.0], dtype=np.float64))

    for (name, identity) in identity_map.items():
        print(f"computation on {name} took {results[identity] * 0.001} milliseconds")

    print("Outputs: ", results)
