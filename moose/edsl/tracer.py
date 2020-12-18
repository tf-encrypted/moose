import inspect
from collections import defaultdict

from moose.compiler.compiler import Compiler
from moose.compiler.host import HostApplyFunctionPass
from moose.compiler.host import NetworkingPass
from moose.compiler.mpspdz import MpspdzApplyFunctionPass
from moose.compiler.pruning import PruningPass
from moose.compiler.replicated.encoding_pass import ReplicatedEncodingPass
from moose.compiler.replicated.lowering_pass import ReplicatedLoweringPass
from moose.compiler.replicated.replicated_pass import ReplicatedOpsPass
from moose.computation.base import Computation
from moose.computation.base import UnknownType
from moose.computation.host import HostPlacement
from moose.computation.host import RunProgramOperation
from moose.computation.mpspdz import MpspdzPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import AddOperation
from moose.computation.standard import ApplyFunctionOperation
from moose.computation.standard import ConstantOperation
from moose.computation.standard import DivOperation
from moose.computation.standard import InputOperation
from moose.computation.standard import LoadOperation
from moose.computation.standard import MulOperation
from moose.computation.standard import OutputOperation
from moose.computation.standard import SaveOperation
from moose.computation.standard import SubOperation
from moose.computation.standard import TensorType
from moose.edsl.base import ApplyFunctionExpression
from moose.edsl.base import ArgumentExpression
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


def trace(abstract_computation, compiler_passes=None, render=False):
    func_signature = inspect.signature(abstract_computation.func)
    symbolic_args = [
        ArgumentExpression(
            arg_name=arg_name,
            datatype=parameter.annotation.datatype,
            placement=parameter.annotation.placement,
            inputs=[],
        )
        for arg_name, parameter in func_signature.parameters.items()
    ]
    expression = abstract_computation.func(*symbolic_args)
    tracer = AstTracer()
    logical_comp = tracer.trace(expression)

    compiler_passes = compiler_passes or [
        MpspdzApplyFunctionPass(),
        HostApplyFunctionPass(),
        ReplicatedEncodingPass(),
        ReplicatedOpsPass(),
        ReplicatedLoweringPass(),
        PruningPass(),
        NetworkingPass(),
    ]
    compiler = Compiler(passes=compiler_passes)
    physical_comp = compiler.run_passes(logical_comp, render=render)

    for op in physical_comp.operations.values():
        get_logger().debug(f"Computation: {op}")
    return physical_comp


class AstTracer:
    def __init__(self):
        self.computation = Computation(operations={}, placements={})
        self.name_counters = defaultdict(int)
        self.operation_cache = dict()
        self.placement_cache = dict()

    def trace(self, expressions: Expression) -> Computation:
        if not isinstance(expressions, (tuple, list)):
            expressions = [expressions]
        for expression in expressions:
            op = self.visit(expression)
            self.computation.add_operation(
                OutputOperation(
                    name=self.get_fresh_name("output"),
                    inputs={"value": op.name},
                    placement_name=op.placement_name,
                )
            )
        return self.computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"

    def visit(self, expression):
        if expression not in self.operation_cache:
            visit_fn = getattr(self, f"visit_{type(expression).__name__}")
            operation = visit_fn(expression)
            self.operation_cache[expression] = operation
        return self.operation_cache[expression]

    def visit_placement_expression(self, placement_expression):
        if placement_expression not in self.placement_cache:
            visit_fn = getattr(self, f"visit_{type(placement_expression).__name__}")
            placement = visit_fn(placement_expression)
            self.placement_cache[placement_expression] = placement
        return self.placement_cache[placement_expression]

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

    def visit_ArgumentExpression(self, argument_expression):
        assert isinstance(argument_expression, ArgumentExpression)
        placement = self.visit_placement_expression(argument_expression.placement)
        output_type = {float: TensorType(datatype="float"), None: UnknownType()}[
            argument_expression.datatype
        ]
        return self.computation.add_operation(
            InputOperation(
                placement_name=placement.name,
                name=argument_expression.arg_name,
                inputs={},
                output_type=output_type,
            )
        )

    def visit_ConstantExpression(self, constant_expression):
        assert isinstance(constant_expression, ConstantExpression)
        placement = self.visit_placement_expression(constant_expression.placement)
        output_type = TensorType(datatype="float")  # TODO use `value` to derive type
        return self.computation.add_operation(
            ConstantOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("constant"),
                output_type=output_type,
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
        # TODO(Morten) we should derive a type from lhs_operation and rhs_operation
        assert lhs_operation.output_type == rhs_operation.output_type
        output_type = lhs_operation.output_type
        return self.computation.add_operation(
            op_type(
                placement_name=placement.name,
                name=self.get_fresh_name(f"{op_name}"),
                inputs={"lhs": lhs_operation.name, "rhs": rhs_operation.name},
                output_type=output_type,
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
        output_type = {float: TensorType(datatype="float"), None: UnknownType()}[
            expression.output_type
        ]
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
        output_type = {float: TensorType(datatype="float"), None: UnknownType()}[
            expression.output_type
        ]
        return self.computation.add_operation(
            RunProgramOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("run_program"),
                path=expression.path,
                args=expression.args,
                inputs=inputs,
                output_type=output_type,
            )
        )
