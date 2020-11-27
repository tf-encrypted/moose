from collections import defaultdict

from moose.compiler.compiler import Compiler
from moose.computation.base import Computation
from moose.computation.base import Graph
from moose.computation.host import RunProgramOperation
from moose.computation.standard import AddOperation
from moose.computation.standard import ApplyFunctionOperation
from moose.computation.standard import ConstantOperation
from moose.computation.standard import DivOperation
from moose.computation.standard import LoadOperation
from moose.computation.standard import MulOperation
from moose.computation.standard import SaveOperation
from moose.computation.standard import SubOperation
from moose.edsl.base import ApplyFunctionExpression
from moose.edsl.base import BinaryOpExpression
from moose.edsl.base import ConstantExpression
from moose.edsl.base import Expression
from moose.edsl.base import LoadExpression
from moose.edsl.base import RunProgramExpression
from moose.edsl.base import SaveExpression
from moose.logger import get_logger


def trace(abstract_computation, *args, render=False, **kwargs):
    expression = abstract_computation.func(*args, **kwargs)
    tracer = AstTracer()
    logical_comp = tracer.trace(expression)
    compiler = Compiler()
    physical_comp = compiler.run_passes(logical_comp, render=render)
    for op in physical_comp.operations():
        get_logger().debug(f"Computation: {op}")
    return physical_comp


class AstTracer:
    def __init__(self):
        self.operations = []
        self.name_counters = defaultdict(int)
        self.known_operations = defaultdict(dict)

    def trace(self, expression: Expression) -> Computation:
        _ = self.visit(expression)
        graph = Graph(nodes={op.name: op for op in self.operations})
        computation = Computation(graph=graph)
        return computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"

    def visit(self, expression):
        if expression not in self.known_operations:
            visit_fn = getattr(self, f"visit_{type(expression).__name__}")
            operation = visit_fn(expression)
            self.operations += [operation]
            self.known_operations[expression] = operation
        return self.known_operations[expression]

    def visit_ConstantExpression(self, constant_expression):
        assert isinstance(constant_expression, ConstantExpression)
        return ConstantOperation(
            placement_name=constant_expression.placement.name,
            name=self.get_fresh_name("constant"),
            value=constant_expression.value,
            inputs={},
        )

    def visit_BinaryOpExpression(self, expression):
        assert isinstance(expression, BinaryOpExpression)
        lhs_expression, rhs_expression = expression.inputs
        lhs_operation = self.visit(lhs_expression)
        rhs_operation = self.visit(rhs_expression)
        op_name = expression.op_name
        op_type = {
            "add": AddOperation,
            "sub": SubOperation,
            "mul": MulOperation,
            "div": DivOperation,
        }[op_name]
        return op_type(
            placement_name=expression.placement.name,
            name=self.get_fresh_name(f"{op_name}"),
            inputs={"lhs": lhs_operation.name, "rhs": rhs_operation.name},
        )

    def visit_LoadExpression(self, load_expression):
        assert isinstance(load_expression, LoadExpression)
        return LoadOperation(
            placement_name=load_expression.placement.name,
            name=self.get_fresh_name("load"),
            key=load_expression.key,
            inputs={},
        )

    def visit_SaveExpression(self, save_expression):
        assert isinstance(save_expression, SaveExpression)
        (value_expression,) = save_expression.inputs
        value_operation = self.visit(value_expression)
        return SaveOperation(
            placement_name=save_expression.placement.name,
            name=self.get_fresh_name("save"),
            key=save_expression.key,
            inputs={"value": value_operation.name},
        )

    def visit_ApplyFunctionExpression(self, expression):
        assert isinstance(expression, ApplyFunctionExpression)
        inputs = {
            f"arg{i}": self.visit(expr).name for i, expr in enumerate(expression.inputs)
        }
        return ApplyFunctionOperation(
            fn=expression.fn,
            placement_name=expression.placement.name,
            name=self.get_fresh_name("apply_function"),
            inputs=inputs,
            output_placements=expression.output_placements,
            output_type=expression.output_type,
        )

    def visit_RunProgramExpression(self, expression):
        assert isinstance(expression, RunProgramExpression)
        inputs = {
            f"arg{i}": self.visit(expr).name for i, expr in enumerate(expression.inputs)
        }
        return RunProgramOperation(
            placement_name=expression.placement.name,
            name=self.get_fresh_name("run_program"),
            path=expression.path,
            args=expression.args,
            inputs=inputs,
        )
