import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose import runtime as rt
from pymoose.computation import utils
from pymoose.logger import get_logger

player0 = pm.host_placement("player0")
player1 = pm.host_placement("player1")
player2 = pm.host_placement("player2")
repl = pm.replicated_placement("replicated", [player0, player1, player2])


class ConcatExample(parameterized.TestCase):
    def _run_concat(self, axis):
        @pm.computation
        def my_comp():
            with player0:
                w = pm.cast(
                    pm.constant(np.array([[1.0, 2.0, 3.0]])), dtype=pm.fixed(14, 23)
                )
                x = pm.cast(
                    pm.constant(np.array([[1.0, 2.0, 3.0]])), dtype=pm.fixed(14, 23)
                )

            with player1:
                y = pm.cast(
                    pm.constant(np.array([[4.0, 5.0, 6.0]])), dtype=pm.fixed(14, 23)
                )

            with player2:
                z = pm.cast(
                    pm.constant(np.array([[7.0, 8.0, 9.0]])), dtype=pm.fixed(14, 23)
                )

            with repl:
                arr = [w, x, y, z]
                result = pm.concatenate(arr, axis=axis)

            with player1:
                result = pm.cast(result, dtype=pm.float64)

            return result

        runtime = rt.LocalMooseRuntime(["player0", "player1", "player2"])
        result = runtime.evaluate_computation(
            computation=my_comp,
            arguments={},
        )
        return result

    @parameterized.parameters(
        (
            0,
            np.array(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            ),
        ),
        (1, np.array([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])),
    )
    def test_concat(self, axis, expected):
        result = self._run_concat(axis)
        val = list(result.values())[0]
        np.testing.assert_equal(val, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concat example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
