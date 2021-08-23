import inspect

from pymoose.computation.standard import UnknownType
from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.deprecated.computation.host import RunProgramOperation
from pymoose.deprecated.computation.mpspdz import MpspdzPlacement
from pymoose.deprecated.computation.standard import ApplyFunctionOperation
from pymoose.deprecated.edsl.base import ApplyFunctionExpression
from pymoose.deprecated.edsl.base import MpspdzPlacementExpression
from pymoose.deprecated.edsl.base import RunProgramExpression
from pymoose.edsl import tracer
from pymoose.edsl.base import ArgumentExpression


def trace(abstract_computation):
    func_signature = inspect.signature(abstract_computation.func)
    symbolic_args = [
        ArgumentExpression(
            arg_name=arg_name,
            vtype=parameter.annotation.vtype,
            placement=parameter.annotation.placement,
            inputs=[],
        )
        for arg_name, parameter in func_signature.parameters.items()
    ]
    expression = abstract_computation.func(*symbolic_args)
    tracer = AstTracer()
    logical_comp = tracer.trace(expression)
    return logical_comp


def trace_and_compile(
    abstract_computation, compiler_passes=None, render=False, ring=64
):
    logical_computation = trace(abstract_computation)
    compiler = Compiler(passes=compiler_passes, ring=ring)
    physical_comp = compiler.compile(logical_computation, render=render)
    return physical_comp


class AstTracer(tracer.AstTracer):
    def visit_MpspdzPlacementExpression(self, mpspdz_placement_expression):
        assert isinstance(mpspdz_placement_expression, MpspdzPlacementExpression)
        player_placements = [
            self.visit_placement_expression(player_placement_expression).name
            for player_placement_expression in mpspdz_placement_expression.players
        ]
        placement = MpspdzPlacement(
            name=mpspdz_placement_expression.name, player_names=player_placements
        )
        return self.computation.add_placement(placement)

    def visit_ApplyFunctionExpression(self, expression):
        assert isinstance(expression, ApplyFunctionExpression)
        inputs = {
            f"arg{i}": self.visit(expr).name for i, expr in enumerate(expression.inputs)
        }
        placement = self.visit_placement_expression(expression.placement)
        output_type = expression.vtype or UnknownType()
        return self.computation.add_operation(
            ApplyFunctionOperation(
                fn=expression.fn,
                placement_name=placement.name,
                name=self.get_fresh_name("apply_function"),
                inputs=inputs,
                output_placements=expression.output_placements,
                output_type=output_type,
            )
        )

    def visit_RunProgramExpression(self, expression):
        assert isinstance(expression, RunProgramExpression)
        inputs = {
            f"arg{i}": self.visit(expr).name for i, expr in enumerate(expression.inputs)
        }
        placement = self.visit_placement_expression(expression.placement)
        return self.computation.add_operation(
            RunProgramOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("run_program"),
                path=expression.path,
                args=expression.args,
                inputs=inputs,
                output_type=expression.vtype,
            )
        )
