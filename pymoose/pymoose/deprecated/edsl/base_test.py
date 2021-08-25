from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation.standard import UnknownType
from pymoose.deprecated import edsl as old_edsl
from pymoose.deprecated.computation import host as host_ops
from pymoose.deprecated.edsl.tracer import trace_and_compile


class EdslTest(parameterized.TestCase):
    def test_run_program(self):
        player0 = edsl.host_placement(name="player0")

        @edsl.computation
        def my_comp():
            x0 = old_edsl.run_program(
                "python",
                ["local_computation.py"],
                edsl.constant(1, placement=player0),
                placement=player0,
            )
            return x0

        concrete_comp = trace_and_compile(my_comp)
        script_py_op = concrete_comp.operation("run_program_0")

        assert script_py_op == host_ops.RunProgramOperation(
            placement_name="player0",
            name="run_program_0",
            inputs={"arg0": "constant_0"},
            path="python",
            args=["local_computation.py"],
            output_type=UnknownType(),
        )
