import inspect
from collections import defaultdict

from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import AbsOperation
from pymoose.computation.standard import AddOperation
from pymoose.computation.standard import AtLeast2DOperation
from pymoose.computation.standard import CastOperation
from pymoose.computation.standard import ConcatenateOperation
from pymoose.computation.standard import ConstantOperation
from pymoose.computation.standard import DecryptOperation
from pymoose.computation.standard import DivOperation
from pymoose.computation.standard import DotOperation
from pymoose.computation.standard import ExpandDimsOperation
from pymoose.computation.standard import ExpOperation
from pymoose.computation.standard import InputOperation
from pymoose.computation.standard import InverseOperation
from pymoose.computation.standard import LessOperation
from pymoose.computation.standard import LoadOperation
from pymoose.computation.standard import MeanOperation
from pymoose.computation.standard import MulOperation
from pymoose.computation.standard import OnesOperation
from pymoose.computation.standard import OutputOperation
from pymoose.computation.standard import ReshapeOperation
from pymoose.computation.standard import SaveOperation
from pymoose.computation.standard import ShapeOperation
from pymoose.computation.standard import SigmoidOperation
from pymoose.computation.standard import SliceOperation
from pymoose.computation.standard import SqueezeOperation
from pymoose.computation.standard import SubOperation
from pymoose.computation.standard import SumOperation
from pymoose.computation.standard import TransposeOperation
from pymoose.computation.standard import UnknownType
from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.edsl.base import AbsExpression
from pymoose.edsl.base import ArgumentExpression
from pymoose.edsl.base import AtLeast2DExpression
from pymoose.edsl.base import BinaryOpExpression
from pymoose.edsl.base import CastExpression
from pymoose.edsl.base import ConcatenateExpression
from pymoose.edsl.base import ConstantExpression
from pymoose.edsl.base import DecryptExpression
from pymoose.edsl.base import ExpandDimsExpression
from pymoose.edsl.base import ExpExpression
from pymoose.edsl.base import Expression
from pymoose.edsl.base import HostPlacementExpression
from pymoose.edsl.base import InverseExpression
from pymoose.edsl.base import LessExpression
from pymoose.edsl.base import LoadExpression
from pymoose.edsl.base import MeanExpression
from pymoose.edsl.base import OnesExpression
from pymoose.edsl.base import ReplicatedPlacementExpression
from pymoose.edsl.base import ReshapeExpression
from pymoose.edsl.base import SaveExpression
from pymoose.edsl.base import ShapeExpression
from pymoose.edsl.base import SigmoidExpression
from pymoose.edsl.base import SliceExpression
from pymoose.edsl.base import SqueezeExpression
from pymoose.edsl.base import SumExpression
from pymoose.edsl.base import TransposeExpression


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
            output_name = self.get_fresh_name("output")
            op = self.visit(expression)
            self.computation.add_operation(
                OutputOperation(
                    name=output_name,
                    inputs={"value": op.name},
                    placement_name=op.placement_name,
                    output_type=op.output_type,
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
        arg_vtype = argument_expression.vtype
        if arg_vtype is None:
            output_type = UnknownType()
        else:
            output_type = arg_vtype
        return self.computation.add_operation(
            InputOperation(
                placement_name=placement.name,
                name=argument_expression.arg_name,
                inputs={},
                output_type=output_type,
            )
        )

    def visit_ConcatenateExpression(self, concatenate_expression):
        assert isinstance(concatenate_expression, ConcatenateExpression)
        arrays = {
            f"array{i}": self.visit(expr).name
            for i, expr in enumerate(concatenate_expression.inputs)
        }
        placement = self.visit_placement_expression(concatenate_expression.placement)
        return self.computation.add_operation(
            ConcatenateOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("concatenate"),
                output_type=concatenate_expression.vtype,
                axis=concatenate_expression.axis,
                inputs=arrays,
            )
        )

    def visit_DecryptExpression(self, decrypt_expression):
        assert isinstance(decrypt_expression, DecryptExpression)
        assert len(decrypt_expression.inputs) == 2
        aes_key_expression, aes_ciphertext_expression = decrypt_expression.inputs
        aes_key_op = self.visit(aes_key_expression)
        aes_ciphertext_op = self.visit(aes_ciphertext_expression)
        placement = self.visit_placement_expression(decrypt_expression.placement)
        return self.computation.add_operation(
            DecryptOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("decrypt"),
                output_type=decrypt_expression.vtype,
                inputs={"key": aes_key_op.name, "ciphertext": aes_ciphertext_op.name},
            )
        )

    def visit_ConstantExpression(self, constant_expression):
        assert isinstance(constant_expression, ConstantExpression)
        placement = self.visit_placement_expression(constant_expression.placement)
        value = constant_expression.value
        vtype = constant_expression.vtype

        if vtype is None:
            output_type = UnknownType()
        else:
            output_type = vtype
        return self.computation.add_operation(
            ConstantOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("constant"),
                output_type=output_type,
                value=value,
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
            "dot": DotOperation,
            "less": LessOperation,
        }[op_name]
        # TODO(Morten) we should derive a type from lhs_operation and rhs_operation
        assert lhs_operation.output_type == rhs_operation.output_type, (
            lhs_operation,
            rhs_operation,
        )
        return self.computation.add_operation(
            op_type(
                placement_name=placement.name,
                name=self.get_fresh_name(f"{op_name}"),
                inputs={"lhs": lhs_operation.name, "rhs": rhs_operation.name},
                output_type=expression.vtype,
            )
        )

    # TODO(Dragos) merge this with visit_BinaryExpression as soon as we have more type information
    # about the BinaryExpression such as input types.
    def visit_LessExpression(self, expression):
        assert isinstance(expression, LessExpression)
        lhs_expression, rhs_expression = expression.inputs
        lhs_operation = self.visit(lhs_expression)
        rhs_operation = self.visit(rhs_expression)
        placement = self.visit_placement_expression(expression.placement)

        assert lhs_operation.output_type == rhs_operation.output_type, (
            lhs_operation,
            rhs_operation,
        )

        return self.computation.add_operation(
            LessOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("less"),
                inputs={"lhs": lhs_operation.name, "rhs": rhs_operation.name},
                output_type=expression.vtype,
                input_type=lhs_operation.output_type,
            )
        )

    def visit_InverseExpression(self, inverse_expression):
        assert isinstance(inverse_expression, InverseExpression)
        (x_expression,) = inverse_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(inverse_expression.placement)
        return self.computation.add_operation(
            InverseOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("inverse"),
                output_type=inverse_expression.vtype,
                inputs={"x": x_operation.name},
            )
        )

    def visit_AbsExpression(self, abs_expression):
        assert isinstance(abs_expression, AbsExpression)
        (x_expression,) = abs_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(abs_expression.placement)
        return self.computation.add_operation(
            AbsOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("abs"),
                output_type=abs_expression.vtype,
                inputs={"x": x_operation.name},
            )
        )

    def visit_CastExpression(self, cast_expression):
        assert isinstance(cast_expression, CastExpression)
        (x_expression,) = cast_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(cast_expression.placement)
        return self.computation.add_operation(
            CastOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("cast"),
                output_type=cast_expression.vtype,
                inputs={"x": x_operation.name},
            )
        )

    def visit_ExpandDimsExpression(self, expand_dims_expression):
        assert isinstance(expand_dims_expression, ExpandDimsExpression)
        (x_expression,) = expand_dims_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(expand_dims_expression.placement)
        return self.computation.add_operation(
            ExpandDimsOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("expand_dims"),
                output_type=expand_dims_expression.vtype,
                axis=expand_dims_expression.axis,
                inputs={"x": x_operation.name},
            )
        )

    def visit_ExpExpression(self, exp_expression):
        assert isinstance(exp_expression, ExpExpression)
        (x_expression,) = exp_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(exp_expression.placement)
        return self.computation.add_operation(
            ExpOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("exp"),
                output_type=exp_expression.vtype,
                inputs={"x": x_operation.name},
            )
        )

    def visit_SigmoidExpression(self, exp_expression):
        assert isinstance(exp_expression, SigmoidExpression)
        (x_expression,) = exp_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(exp_expression.placement)
        return self.computation.add_operation(
            SigmoidOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("sigmoid"),
                output_type=exp_expression.vtype,
                inputs={"x": x_operation.name},
            )
        )

    def visit_SqueezeExpression(self, squeeze_expression):
        assert isinstance(squeeze_expression, SqueezeExpression)
        (x_expression,) = squeeze_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(squeeze_expression.placement)
        return self.computation.add_operation(
            SqueezeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("squeeze"),
                output_type=squeeze_expression.vtype,
                axis=squeeze_expression.axis,
                inputs={"x": x_operation.name},
            )
        )

    def visit_OnesExpression(self, ones_expression):
        assert isinstance(ones_expression, OnesExpression)
        (shape_expression,) = ones_expression.inputs
        shape_operation = self.visit(shape_expression)
        placement = self.visit_placement_expression(ones_expression.placement)
        dtype = ones_expression.dtype
        return self.computation.add_operation(
            OnesOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("ones"),
                output_type=ones_expression.vtype,
                dtype=dtype,
                inputs={"shape": shape_operation.name},
            )
        )

    def visit_SumExpression(self, sum_expression):
        assert isinstance(sum_expression, SumExpression)
        (x_expression,) = sum_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(sum_expression.placement)
        return self.computation.add_operation(
            SumOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("sum"),
                output_type=sum_expression.vtype,
                axis=sum_expression.axis,
                inputs={"x": x_operation.name},
            )
        )

    def visit_MeanExpression(self, mean_expression):
        assert isinstance(mean_expression, MeanExpression)
        (x_expression,) = mean_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(mean_expression.placement)
        return self.computation.add_operation(
            MeanOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("mean"),
                output_type=mean_expression.vtype,
                axis=mean_expression.axis,
                inputs={"x": x_operation.name},
            )
        )

    def visit_TransposeExpression(self, transpose_expression):
        assert isinstance(transpose_expression, TransposeExpression)
        (x_expression,) = transpose_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(transpose_expression.placement)
        return self.computation.add_operation(
            TransposeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("transpose"),
                output_type=transpose_expression.vtype,
                axes=transpose_expression.axes,
                inputs={"x": x_operation.name},
            )
        )

    def visit_ReshapeExpression(self, reshape_expression):
        assert isinstance(reshape_expression, ReshapeExpression)
        (x_expression, shape_expression) = reshape_expression.inputs
        x_operation = self.visit(x_expression)
        shape_operation = self.visit(shape_expression)
        placement = self.visit_placement_expression(reshape_expression.placement)
        return self.computation.add_operation(
            ReshapeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("reshape"),
                output_type=reshape_expression.vtype,
                inputs={"x": x_operation.name, "shape": shape_operation.name},
            )
        )

    def visit_AtLeast2DExpression(self, atleast_2d_expression):
        assert isinstance(atleast_2d_expression, AtLeast2DExpression)
        (x_expression,) = atleast_2d_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(atleast_2d_expression.placement)
        return self.computation.add_operation(
            AtLeast2DOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("atleast_2d"),
                output_type=atleast_2d_expression.vtype,
                to_column_vector=atleast_2d_expression.to_column_vector,
                inputs={"x": x_operation.name},
            )
        )

    def visit_SliceExpression(self, slice_expression):
        assert isinstance(slice_expression, SliceExpression)
        (x_expression,) = slice_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(slice_expression.placement)
        return self.computation.add_operation(
            SliceOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("slice"),
                output_type=slice_expression.vtype,
                begin=slice_expression.begin,
                end=slice_expression.end,
                inputs={"x": x_operation.name},
            )
        )

    def visit_ShapeExpression(self, shape_expression):
        assert isinstance(shape_expression, ShapeExpression)
        (x_expression,) = shape_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(shape_expression.placement)
        return self.computation.add_operation(
            ShapeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("shape"),
                output_type=shape_expression.vtype,
                inputs={"x": x_operation.name},
            )
        )

    def visit_LoadExpression(self, load_expression):
        assert isinstance(load_expression, LoadExpression)
        key_expression, query_expression = load_expression.inputs
        key_operation = self.visit(key_expression)
        query_operation = self.visit(query_expression)
        placement = self.visit_placement_expression(load_expression.placement)
        output_type = load_expression.vtype or UnknownType()
        return self.computation.add_operation(
            LoadOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("load"),
                inputs={"key": key_operation.name, "query": query_operation.name},
                output_type=output_type,
            )
        )

    def visit_SaveExpression(self, save_expression):
        assert isinstance(save_expression, SaveExpression)
        (key_expression, value_expression) = save_expression.inputs
        key_operation = self.visit(key_expression)
        value_operation = self.visit(value_expression)
        placement = self.visit_placement_expression(save_expression.placement)
        return self.computation.add_operation(
            SaveOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("save"),
                inputs={"key": key_operation.name, "value": value_operation.name},
            )
        )
