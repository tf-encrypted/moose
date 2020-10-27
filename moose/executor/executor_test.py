import logging
import pathlib
import unittest

from absl.testing import parameterized

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import add
from moose.compiler.edsl import computation
from moose.compiler.edsl import constant
from moose.compiler.edsl import div
from moose.compiler.edsl import function
from moose.compiler.edsl import mul
from moose.compiler.edsl import run_program
from moose.compiler.edsl import save
from moose.compiler.edsl import sub
from moose.logger import get_logger
from moose.runtime import TestRuntime as Runtime

get_logger().setLevel(level=logging.DEBUG)


def _create_test_players(number_of_players=2):
    return [HostPlacement(name=f"player_{i}") for i in range(number_of_players)]


def _run_computation(comp, players):
    runtime = Runtime([worker.name for worker in players])
    placement_assignment = {
        player: runtime.executors[player.name] for player in players
    }
    concrete_comp = comp.trace_func()
    runtime.evaluate_computation(
        concrete_comp, placement_assignment=placement_assignment
    )
    computation_result = runtime.executors[players[-1].name].store
    return computation_result


class ExecutorTest(parameterized.TestCase):
    def test_call_python_function(self):
        player0, player1 = _create_test_players(2)

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            with player0:
                out = add_one(constant(3))
            with player1:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, [player0, player1])
        self.assertEqual(comp_result["result"], 4)

    def test_constant(self):
        player0, player1 = _create_test_players(2)

        @computation
        def my_comp():
            with player0:
                out = constant(5)
            with player1:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, [player0, player1])
        self.assertEqual(comp_result["result"], 5)

    @parameterized.parameters(
        {"op": op, "expected_result": expected_result}
        for (op, expected_result) in zip([add, sub, mul, div], [7, 3, 10, 2.5])
    )
    def test_op(self, op, expected_result):
        player0, player1 = _create_test_players(2)

        @computation
        def my_comp():
            with player0:
                out = op(constant(5), constant(2))
            with player1:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, [player0, player1])
        self.assertEqual(comp_result["result"], expected_result)

    def test_run_program(self):
        player0, player1, player2 = _create_test_players(3)
        test_fixtures_file = str(
            pathlib.Path(__file__)
            .parent.absolute()
            .joinpath("executor_test_fixtures.py")
        )

        @computation
        def my_comp():
            with player0:
                c0 = constant(3)
                c1 = constant(2)
            with player1:
                out = run_program("python", [test_fixtures_file], c0, c1)
            with player2:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, [player0, player1, player2])
        self.assertEqual(comp_result["result"], 6)


if __name__ == "__main__":
    unittest.main()
