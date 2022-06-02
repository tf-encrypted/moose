import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose import runtime as rt
from pymoose.logger import get_logger

player0 = pm.host_placement("player0")
player1 = pm.host_placement("player1")
player2 = pm.host_placement("player2")
repl = pm.replicated_placement("replicated", [player0, player1, player2])


class AddNExample(parameterized.TestCase):
    def _run_add_n(self):
        @pm.computation
        def my_comp():
            with player0:
                w = pm.cast(
                    pm.constant(np.array([1.0, 2.0, 3.0])), dtype=pm.fixed(14, 23)
                )
                x = pm.cast(
                    pm.constant(np.array([1.0, 2.0, 3.0])), dtype=pm.fixed(14, 23)
                )

            with player1:
                y = pm.cast(
                    pm.constant(np.array([4.0, 5.0, 6.0])), dtype=pm.fixed(14, 23)
                )

            with player2:
                z = pm.cast(
                    pm.constant(np.array([7.0, 8.0, 9.0])), dtype=pm.fixed(14, 23)
                )

            with repl:
                arr = [w, x, y, z]
                result = pm.add_n(arr)

            with player1:
                result = pm.cast(result, dtype=pm.float64)

            return result

        runtime = rt.LocalMooseRuntime(["player0", "player1", "player2"])
        result = runtime.evaluate_computation(
            computation=my_comp,
            arguments={},
        )
        return result

    def test_add_n(self):
        result = self._run_add_n()
        val = list(result.values())[0]
        assert all(val == np.array([13.0, 17.0, 21.0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add_n example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
