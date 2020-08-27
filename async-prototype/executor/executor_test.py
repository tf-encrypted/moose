import logging
import os
import unittest

from absl.testing import parameterized

from edsl import Role
from edsl import add
from edsl import computation
from edsl import constant
from edsl import div
from edsl import function
from edsl import mul
from edsl import run_python_script
from edsl import save
from edsl import sub
from logger import get_logger
from runtime import TestRuntime

get_logger().setLevel(level=logging.DEBUG)


def _create_test_players(number_of_players=2):
    return [Role(name=f"player_{i}") for i in range(number_of_players)]


def _run_computation(comp, players):
    runtime = TestRuntime(num_workers=len(players))
    role_assignment = {players[i]: runtime.executors[i] for i in range(len(players))}
    concrete_comp = comp.trace_func()
    runtime.evaluate_computation(concrete_comp, role_assignment=role_assignment)
    computation_result = runtime.executors[-1].get_store()
    return computation_result


class ExecutorTest(parameterized.TestCase):
    def test_constant(self):
        players = _create_test_players(2)

        @computation
        def my_comp():
            with players[0]:
                out = constant(5)
            with players[1]:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, players)
        self.assertEqual(comp_result["result"], 5)

    @parameterized.parameters(
        {"op": op, "expected_result": expected_result}
        for (op, expected_result) in zip([add, sub, mul, div], [7, 3, 10, 2.5])
    )
    def test_op(self, op, expected_result):
        players = _create_test_players(2)

        @computation
        def my_comp():
            with players[0]:
                out = op(constant(5), constant(2))
            with players[1]:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, players)
        self.assertEqual(comp_result["result"], expected_result)

    def test_call_python_function(self):
        players = _create_test_players(2)

        @function
        def add_one(x):
            return x + 1

        @computation
        def my_comp():
            with players[0]:
                out = add_one(constant(3))
            with players[1]:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, players)
        self.assertEqual(comp_result["result"], 4)

    def test_run_python_script(self):
        players = _create_test_players(3)

        @computation
        def my_comp():
            with players[0]:
                c0 = constant(3)
                c1 = constant(2)
            with players[1]:
                out = run_python_script(
                    os.getcwd() + "/examples/local_computation.py", c0, c1
                )
            with players[1]:
                res = save(out, "result")
            return res

        comp_result = _run_computation(my_comp, players)
        self.assertEqual(comp_result["result"], 6)


if __name__ == "__main__":
    unittest.main()
