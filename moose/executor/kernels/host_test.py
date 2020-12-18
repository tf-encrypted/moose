import logging
import pathlib
import unittest

from absl.testing import parameterized

from moose.edsl.base import computation
from moose.edsl.base import constant
from moose.edsl.base import function
from moose.edsl.base import host_placement
from moose.edsl.base import run_program
from moose.edsl.base import save
from moose.edsl.tracer import trace
from moose.logger import get_logger
from moose.runtime import TestRuntime as Runtime

get_logger().setLevel(level=logging.DEBUG)


def _create_test_players(number_of_players=2):
    return [host_placement(name=f"player_{i}") for i in range(number_of_players)]


def _run_computation(comp, players):
    runtime = Runtime()
    placement_instantiation = {player: player.name for player in players}
    concrete_comp = trace(comp)
    runtime.evaluate_computation(
        concrete_comp, placement_instantiation=placement_instantiation
    )
    return runtime.get_executor(players[-1].name).store


class HostKernelTest(parameterized.TestCase):
    def test_call_python_function(self):
        player0, player1 = _create_test_players(2)

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            out = add_one(constant(3, placement=player0), placement=player0)
            res = save(out, "result", placement=player1)
            return res

        comp_result = _run_computation(my_comp, [player0, player1])
        self.assertEqual(comp_result["result"], 4)

    def test_run_program(self):
        player0, player1, player2 = _create_test_players(3)
        test_fixtures_file = str(
            pathlib.Path(__file__).parent.absolute().joinpath("host_test_fixtures.py")
        )

        @computation
        def my_comp():
            c0 = constant(3, placement=player0)
            c1 = constant(2, placement=player0)
            out = run_program("python", [test_fixtures_file], c0, c1, placement=player1)
            res = save(out, "result", placement=player2)
            return res

        comp_result = _run_computation(my_comp, [player0, player1, player2])
        self.assertEqual(comp_result["result"], 6)


if __name__ == "__main__":
    unittest.main()
