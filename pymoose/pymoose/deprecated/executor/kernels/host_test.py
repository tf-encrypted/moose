import pathlib
import unittest

from absl.testing import parameterized

from pymoose.computation import dtypes
from pymoose.computation.standard import TensorType
from pymoose.deprecated import edsl as old_edsl
from pymoose.deprecated.edsl.tracer import trace_and_compile
from pymoose.deprecated.testing import run_test_computation
from pymoose.edsl import base as edsl


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
            out = old_edsl.run_program(
                "python",
                [test_fixtures_file],
                c0,
                c1,
                placement=player1,
                vtype=TensorType(dtype=dtypes.int64),
            )
            res = edsl.save("result", out, placement=player2)
            return res

        comp_result = run_test_computation(
            trace_and_compile(my_comp), [player0, player1, player2]
        )
        self.assertEqual(comp_result[player2]["result"], 6)


if __name__ == "__main__":
    unittest.main()
