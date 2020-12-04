from collections import defaultdict

from moose.compiler.compiler import Compiler
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.host import RunProgramOperation
from moose.computation.mpspdz import MpspdzPlacement
from moose.computation.replicated import ReplicatedPlacement
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
from moose.edsl.base import HostPlacementExpression
from moose.edsl.base import LoadExpression
from moose.edsl.base import MpspdzPlacementExpression
from moose.edsl.base import ReplicatedPlacementExpression
from moose.edsl.base import RunProgramExpression
from moose.edsl.base import SaveExpression
from moose.logger import get_logger


def trace(abstract_computation, *args, compiler_passes=None, render=False, **kwargs):
    expression = abstract_computation.func(*args, **kwargs)
    tracer = AstTracer()
    logical_comp = tracer.trace(expression)
    compiler = Compiler(passes=compiler_passes)
    physical_comp = compiler.run_passes(logical_comp, render=render)
    for op in physical_comp.operations.values():
        get_logger().debug(f"Computation: {op}")
    return physical_comp


class AstTracer:
    def __init__(self):
        self.computation = Computation(operations={}, placements={})
        self.name_counters = defaultdict(int)
        self.known_operations = dict()
        self.known_placements = dict()

    def trace(self, expression: Expression) -> Computation:
        _ = self.visit(expression)
        return self.computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"

    def visit(self, expression):
        if expression not in self.known_operations:
            visit_fn = getattr(self, f"visit_{type(expression).__name__}")
            operation = visit_fn(expression)
            self.known_operations[expression] = operation
        return self.known_operations[expression]

    def visit_placement_expression(self, placement_expression):
        if placement_expression not in self.known_placements:
            visit_fn = getattr(self, f"visit_{type(placement_expression).__name__}")
            placement = visit_fn(placement_expression)
            self.known_placements[placement_expression] = placement
        return self.known_placements[placement_expression]

    def visit_HostPlacementExpression(self, host_placement_expression):
        assert isinstance(host_placement_expression, HostPlacementExpression)
        placement = HostPlacement(name=host_placement_expression.name)
        return self.computation.add_placement(placement)

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

    def visit_ReplicatedPlacementExpression(self, replicated_placement_expression):
        assert isinstance(
            replicated_placement_expression, ReplicatedPlacementExpression
        )
        player_placements = [
            self.visit_placement_expression(player_placement_expression).name
            for player_placement_expression in replicated_placement_expression.players
        ]
        placement = ReplicatedPlacement(
            name=replicated_placement_expression.name, player_names=player_placements
        )
        return self.computation.add_placement(placement)

    def visit_ConstantExpression(self, constant_expression):
        assert isinstance(constant_expression, ConstantExpression)
        placement = self.visit_placement_expression(constant_expression.placement)
        return self.computation.add_operation(
            ConstantOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("constant"),
                value=constant_expression.value,
                inputs={},
            )
        )

    def visit_BinaryOpExpression(self, expression):
        assert isinstance(expression, BinaryOpExpression)
        lhs_expression, rhs_expression = expression.inputs
        lhs_operation = self.visit(lhs_expression)
        rhs_operation = self.visit(rhs_expression)
        placement = self.visit_placement_expression(expression.placement)
        op_name = expression.op_name
        op_type = {
            "add": AddOperation,
            "sub": SubOperation,
            "mul": MulOperation,
            "div": DivOperation,
        }[op_name]
        return self.computation.add_operation(
            op_type(
                placement_name=placement.name,
                name=self.get_fresh_name(f"{op_name}"),
                inputs={"lhs": lhs_operation.name, "rhs": rhs_operation.name},
            )
        )

    def visit_LoadExpression(self, load_expression):
        assert isinstance(load_expression, LoadExpression)
        placement = self.visit_placement_expression(load_expression.placement)
        return self.computation.add_operation(
            LoadOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("load"),
                key=load_expression.key,
                inputs={},
            )
        )

    def visit_SaveExpression(self, save_expression):
        assert isinstance(save_expression, SaveExpression)
        (value_expression,) = save_expression.inputs
        value_operation = self.visit(value_expression)
        placement = self.visit_placement_expression(save_expression.placement)
        return self.computation.add_operation(
            SaveOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("save"),
                key=save_expression.key,
                inputs={"value": value_operation.name},
            )
        )

    def visit_ApplyFunctionExpression(self, expression):
        assert isinstance(expression, ApplyFunctionExpression)
        inputs = {
            f"arg{i}": self.visit(expr).name for i, expr in enumerate(expression.inputs)
        }
        placement = self.visit_placement_expression(expression.placement)
        return self.computation.add_operation(
            ApplyFunctionOperation(
                fn=expression.fn,
                placement_name=placement.name,
                name=self.get_fresh_name("apply_function"),
                inputs=inputs,
                output_placements=expression.output_placements,
                output_type=expression.output_type,
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
            )
        )
