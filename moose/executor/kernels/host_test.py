import pathlib
import unittest

from absl.testing import parameterized

from moose.computation import dtypes
from moose.computation.standard import TensorType
from moose.edsl import base as edsl
from moose.edsl.tracer import trace
from moose.testing import run_test_computation


class HostKernelTest(parameterized.TestCase):
    def test_run_program(self):
        player0 = edsl.host_placement("player0")
        player1 = edsl.host_placement("player1")
        player2 = edsl.host_placement("player2")

        test_fixtures_file = str(
            pathlib.Path(__file__).parent.absolute().joinpath("host_test_fixtures.py")
        )

        @edsl.computation
        def my_comp():
            c0 = edsl.constant(3, placement=player0)
            c1 = edsl.constant(2, placement=player0)
            out = edsl.run_program(
                "python",
                [test_fixtures_file],
                c0,
                c1,
                placement=player1,
                output_type=TensorType(dtype=dtypes.int64),
            )
            res = edsl.save("result", out, placement=player2)
            return res

        comp_result = run_test_computation(trace(my_comp), [player0, player1, player2])
        self.assertEqual(comp_result[player2]["result"], 6)


if __name__ == "__main__":
    unittest.main()
