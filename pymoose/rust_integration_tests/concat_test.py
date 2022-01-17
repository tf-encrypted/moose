import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime

player0 = edsl.host_placement("player0")
player1 = edsl.host_placement("player1")
player2 = edsl.host_placement("player2")
repl = edsl.replicated_placement("replicated", [player0, player1, player2])


class ConcatExample(parameterized.TestCase):
    def _run_concat(self):
        @edsl.computation
        def my_comp():
            with player0:
                w = edsl.cast(
                    edsl.constant(np.array([1.0, 2.0, 3.0])), dtype=edsl.fixed(14, 23)
                )
                x = edsl.cast(
                    edsl.constant(np.array([1.0, 2.0, 3.0])), dtype=edsl.fixed(14, 23)
                )

            with player1:
                y = edsl.cast(
                    edsl.constant(np.array([4.0, 5.0, 6.0])), dtype=edsl.fixed(14, 23)
                )

            with player2:
                z = edsl.cast(
                    edsl.constant(np.array([7.0, 8.0, 9.0])), dtype=edsl.fixed(14, 23)
                )

            with repl:
                arr = [w, x, y, z]
                result = edsl.concatenate(arr)

            with player1:
                result = edsl.cast(result, dtype=edsl.float64)

            return result

        executors_storage = {
            "player0": {},
            "player1": {},
            "player2": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=executors_storage)
        concrete_comp = edsl.trace(my_comp)

        comp_bin = utils.serialize_computation(concrete_comp)
        rust_compiled = elk_compiler.compile_computation(
            comp_bin, ["typing", "full", "prune", "networking", "typing", "toposort"]
        )

        result = runtime.evaluate_compiled(
            comp_bin=rust_compiled,
            role_assignment={
                "player0": "player0",
                "player1": "player1",
                "player2": "player2",
            },
            arguments={},
        )
        return result

    def test_concat(self):
        result = self._run_concat()
        val = list(result.values())[0]
        assert all(
            val
            == np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="concat example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
