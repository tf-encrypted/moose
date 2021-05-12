import argparse
import logging
import unittest

import numpy as np

from moose import edsl
from moose.logger import get_logger
from moose.testing import TestRuntime as Runtime

from moose.computation import dtypes
from moose.testing import run_test_computation

alice = edsl.host_placement(name="alice")
bob = edsl.host_placement(name="bob")
carole = edsl.host_placement(name="carole")
dave = edsl.host_placement(name="dave")
rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

@edsl.computation
def my_comp():

    with alice:
        x = edsl.constant(np.array([1, 2], dtype=np.float64))
        x = edsl.cast(x, dtype=dtypes.fixed(30, 2))
        # # sint(), sfix()
        # # x = edsl.cast(x, dtype=dtypes.fixed(8, 27))

    with bob:
        y = edsl.constant(np.array([1, 1], dtype=np.float64))
        # y = edsl.cast(y, dtype=dtypes.uint64)
        y = edsl.cast(y, dtype=dtypes.fixed(30, 2))

    with rep:
        z = edsl.mul(x, y)

    with dave:
        # z = edsl.cast(z, dtype=dtypes.uint64)
        z = edsl.cast(z, dtype=dtypes.float64)
        res_dave = edsl.save("res", z)

    return (res_dave)

concrete_comp = edsl.tracer.trace_and_compile(my_comp)
results = run_test_computation(concrete_comp, [alice, bob, carole, dave])

print("Done")
print(results[dave]["res"])

