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
from moose.runtime import run_test_computation


class HostKernelTest(parameterized.TestCase):
    def test_call_python_function(self):
        player0 = host_placement("player0")
        player1 = host_placement("player1")

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            out = add_one(constant(3, placement=player0), placement=player0)
            res = save("result", out, placement=player1)
            return res

        comp_result = run_test_computation(trace(my_comp), [player0, player1])
        self.assertEqual(comp_result[player1]["result"], 4)

    def test_run_program(self):
        player0 = host_placement("player0")
        player1 = host_placement("player1")
        player2 = host_placement("player2")

        test_fixtures_file = str(
            pathlib.Path(__file__).parent.absolute().joinpath("host_test_fixtures.py")
        )

        @computation
        def my_comp():
            c0 = constant(3, placement=player0)
            c1 = constant(2, placement=player0)
            out = run_program("python", [test_fixtures_file], c0, c1, placement=player1)
            res = save("result", out, placement=player2)
            return res

        comp_result = run_test_computation(trace(my_comp), [player0, player1, player2])
        self.assertEqual(comp_result[player2]["result"], 6)


if __name__ == "__main__":
    unittest.main()
