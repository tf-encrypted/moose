import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class MirroredOpsExample(parameterized.TestCase):
    def _setup_comp(self):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        mir3 = pm.mirrored_placement(name="mir3", players=[alice, bob, carole])

        @pm.computation
        def my_comp():
            with mir3:
                x = pm.constant(np.array([1.5, 2.3, 3, 3], dtype=np.float64))
                x = pm.cast(x, dtype=pm.fixed(8, 27))
            with alice:
                y = pm.cast(x, dtype=pm.float64)

            return y

        return my_comp

    def test_example_execute(self):
        comp = self._setup_comp()
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        result_dict = runtime.evaluate_computation(
            computation=comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )

        actual_result = list(result_dict.values())[0]

        np.testing.assert_almost_equal(actual_result[0], 1.5)
        np.testing.assert_almost_equal(actual_result[1], 2.3)
        np.testing.assert_almost_equal(actual_result[2], 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirrored placement example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
