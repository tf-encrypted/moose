import argparse
import logging

import numpy as np

import pymoose as pm
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import GrpcMooseRuntime

@pm.computation
def my_computation():
    alice = pm.host_placement("alice")
    bob = pm.host_placement("bob")
    carole = pm.host_placement("carole")

    with alice:
        x = pm.constant(np.array([1., 2.], dtype=np.float64))
    
    with bob:
        y = pm.constant(np.array([1., 2.], dtype=np.float64))

    with carole:
        z = pm.add(x, y)

    return z

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    runtime = GrpcMooseRuntime({ 
        "alice": "localhost:50000",
        "bob": "localhost:50001",
        "carole": "localhost:50002"
     })

    comp = pm.trace(my_computation)
    results = runtime.evaluate_computation(comp)
    print(results)
