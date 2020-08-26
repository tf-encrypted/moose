import logging
import unittest

from computation import Computation
from edsl import Role
from edsl import computation
from edsl import constant
from edsl import load
from edsl import save
from logger import get_logger
from logger import set_logger
from runtime import RemoteRuntime
from runtime import TestRuntime

get_logger().setLevel(level=logging.DEBUG)


def create_test_players(number_of_players=2):
    return [Role(name=f"player_{i}") for i in range(number_of_players)]


def create_test_runtimes(players):
    runtime = TestRuntime(num_workers=len(players))
    role_assignment = {players[i]: runtime.executors[i] for i in range(len(players))}
    return runtime, role_assignment

def create_op_computation(players, op, *args):
    @computation
    def my_comp():
        with players[0]:
            out = op(*args)
        with players[1]:
            res = save(out, "result")
        return res
    return my_comp

class ExecutorTest(unittest.TestCase):
    def test_constant(self):

        players = create_test_players(2)
        runtime, role_assignment = create_test_runtimes(players)
        
        my_comp = create_op_computation(players, constant, 5)

        concrete_comp = my_comp.trace_func()
        runtime.evaluate_computation(concrete_comp, role_assignment=role_assignment)
        computation_result = runtime.executors[1].get_store()

        self.assertEqual(computation_result["result"], 5)


if __name__ == "__main__":
    unittest.main()
