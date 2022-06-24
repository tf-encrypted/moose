import argparse
import logging
import pathlib

import numpy as np

import pymoose as pm
from pymoose.logger import get_logger

_DATA_DIR = pathlib.Path(__file__).parent

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
carole = pm.host_placement("carole")


@pm.computation
def my_computation(
    v: pm.Argument(placement=alice, vtype=pm.TensorType(pm.float64)),
):
    with alice:
        x = pm.load(str((_DATA_DIR / "x.npy").resolve()), dtype=pm.float64)

    with bob:
        y = pm.load(str((_DATA_DIR / "y.npy").resolve()), dtype=pm.float64)
        t = pm.constant(np.array([5.0, 6.0], dtype=np.float64))

    with carole:
        z = pm.add(x, y)
        w = pm.add(z, v)
        o = pm.add(w, t)

    return o


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    role_map = {
        alice: "localhost:50000",
        bob: "localhost:50001",
        carole: "localhost:50002",
    }

    runtime = pm.GrpcMooseRuntime(role_map)
    runtime.set_default()

    results, _ = my_computation(np.array([5.0, 6.0], dtype=np.float64))

    print("Outputs: ", results)
