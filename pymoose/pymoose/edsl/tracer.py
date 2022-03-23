import inspect
from collections import defaultdict

from pymoose.computation import utils
from pymoose.computation.base import Computation
from pymoose.computation.base import OpSignature
from pymoose.computation.host import HostPlacement
from pymoose.computation.logical import AbsOperation
from pymoose.computation.logical import AddNOperation
from pymoose.computation.logical import AddOperation
from pymoose.computation.logical import ArgmaxOperation
from pymoose.computation.logical import AtLeast2DOperation
from pymoose.computation.logical import BitwiseOrOperation
from pymoose.computation.logical import CastOperation
from pymoose.computation.logical import ConcatenateOperation
from pymoose.computation.logical import ConstantOperation
from pymoose.computation.logical import DecryptOperation
from pymoose.computation.logical import DivOperation
from pymoose.computation.logical import DotOperation
from pymoose.computation.logical import ExpandDimsOperation
from pymoose.computation.logical import ExpOperation
from pymoose.computation.logical import IdentityOperation
from pymoose.computation.logical import IndexAxisOperation
from pymoose.computation.logical import InputOperation
from pymoose.computation.logical import InverseOperation
from pymoose.computation.logical import LessOperation
from pymoose.computation.logical import LoadOperation
from pymoose.computation.logical import Log2Operation
from pymoose.computation.logical import LogOperation
from pymoose.computation.logical import MaximumOperation
from pymoose.computation.logical import MeanOperation
from pymoose.computation.logical import MulOperation
from pymoose.computation.logical import MuxOperation
from pymoose.computation.logical import OnesOperation
from pymoose.computation.logical import OutputOperation
from pymoose.computation.logical import ReshapeOperation
from pymoose.computation.logical import SaveOperation
from pymoose.computation.logical import ShapeOperation
from pymoose.computation.logical import SigmoidOperation
from pymoose.computation.logical import SliceOperation
from pymoose.computation.logical import SoftmaxOperation
from pymoose.computation.logical import SqueezeOperation
from pymoose.computation.logical import SubOperation
from pymoose.computation.logical import SumOperation
from pymoose.computation.logical import TransposeOperation
from pymoose.computation.logical import UnitType
from pymoose.computation.logical import UnknownType
from pymoose.computation.mirrored import MirroredPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.edsl.base import AbsExpression
from pymoose.edsl.base import AddNExpression
from pymoose.edsl.base import ArgmaxExpression
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
from pymoose.edsl.base import IdentityExpression
from pymoose.edsl.base import IndexAxisExpression
from pymoose.edsl.base import InverseExpression
from pymoose.edsl.base import LoadExpression
from pymoose.edsl.base import Log2Expression
from pymoose.edsl.base import LogExpression
from pymoose.edsl.base import MaximumExpression
from pymoose.edsl.base import MeanExpression
from pymoose.edsl.base import MirroredPlacementExpression
from pymoose.edsl.base import MuxExpression
from pymoose.edsl.base import OnesExpression
from pymoose.edsl.base import ReplicatedPlacementExpression
from pymoose.edsl.base import ReshapeExpression
from pymoose.edsl.base import SaveExpression
from pymoose.edsl.base import ShapeExpression
from pymoose.edsl.base import SigmoidExpression
from pymoose.edsl.base import SliceExpression
from pymoose.edsl.base import SoftmaxExpression
from pymoose.edsl.base import SqueezeExpression
from pymoose.edsl.base import SumExpression
from pymoose.edsl.base import TransposeExpression
from pymoose.rust import elk_compiler


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


def trace_and_compile(abstract_computation, compiler_passes=None):
    logical_computation = trace(abstract_computation)
    comp_bin = utils.serialize_computation(logical_computation)
    physical_comp_ref = elk_compiler.compile_computation(comp_bin, compiler_passes)
    return physical_comp_ref


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
                    signature=OpSignature(
                        input_types={"value": op.return_type},
                        return_type=op.return_type,
                    ),
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

    def visit_MirroredPlacementExpression(self, mirrored_placement_expression):
        assert isinstance(mirrored_placement_expression, MirroredPlacementExpression)
        player_placements = [
            self.visit_placement_expression(player_placement_expression).name
            for player_placement_expression in mirrored_placement_expression.players
        ]
        placement = MirroredPlacement(
            name=mirrored_placement_expression.name, player_names=player_placements
        )
        return self.computation.add_placement(placement)

    def visit_IdentityExpression(self, identity_expression):
        assert isinstance(identity_expression, IdentityExpression)
        placement = self.visit_placement_expression(identity_expression.placement)
        input_expression = identity_expression.inputs[0]
        input_op = self.visit(input_expression)
        input_type = input_op.return_type
        output_type = input_expression.vtype
        return self.computation.add_operation(
            IdentityOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("identity"),
                inputs={"x": input_op.name},
                signature=OpSignature(
                    input_types={"x": input_type}, return_type=output_type
                ),
            )
        )

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
                signature=OpSignature(input_types={}, return_type=output_type,),
            )
        )

    def visit_AddNExpression(self, add_n_expression):
        assert isinstance(add_n_expression, AddNExpression)
        array_inputs, array_types = {}, {}
        for i, expr in enumerate(add_n_expression.inputs):
            array_op = self.visit(expr)
            array_inputs[f"array{i}"] = array_op.name
            array_types[f"array{i}"] = array_op.return_type

        placement = self.visit_placement_expression(add_n_expression.placement)
        return self.computation.add_operation(
            AddNOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("add_n"),
                inputs=array_inputs,
                signature=OpSignature(
                    input_types=array_types, return_type=add_n_expression.vtype
                ),
            )
        )

    def visit_ConcatenateExpression(self, concatenate_expression):
        assert isinstance(concatenate_expression, ConcatenateExpression)
        array_inputs, array_types = {}, {}
        for i, expr in enumerate(concatenate_expression.inputs):
            array_op = self.visit(expr)
            array_inputs[f"array{i}"] = array_op.name
            array_types[f"array{i}"] = array_op.return_type

        placement = self.visit_placement_expression(concatenate_expression.placement)
        return self.computation.add_operation(
            ConcatenateOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("concatenate"),
                axis=concatenate_expression.axis,
                inputs=array_inputs,
                signature=OpSignature(
                    input_types=array_types, return_type=concatenate_expression.vtype,
                ),
            )
        )

    def visit_MaximumExpression(self, maximum_expression):
        assert isinstance(maximum_expression, MaximumExpression)
        array_inputs, array_types = {}, {}
        for i, expr in enumerate(maximum_expression.inputs):
            array_op = self.visit(expr)
            array_inputs[f"array{i}"] = array_op.name
            array_types[f"array{i}"] = array_op.return_type

        placement = self.visit_placement_expression(maximum_expression.placement)
        return self.computation.add_operation(
            MaximumOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("maximum"),
                inputs=array_inputs,
                signature=OpSignature(
                    input_types=array_types, return_type=maximum_expression.vtype,
                ),
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
                inputs={"key": aes_key_op.name, "ciphertext": aes_ciphertext_op.name},
                signature=OpSignature(
                    input_types={
                        "key": aes_key_op.return_type,
                        "ciphertext": aes_ciphertext_op.return_type,
                    },
                    return_type=decrypt_expression.vtype,
                ),
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
                value=value,
                inputs={},
                signature=OpSignature(input_types={}, return_type=output_type),
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
            "or": BitwiseOrOperation,
            "less": LessOperation,
        }[op_name]
        lhs_type = lhs_operation.return_type
        rhs_type = rhs_operation.return_type
        assert lhs_type == rhs_type, (
            lhs_operation,
            rhs_operation,
        )
        return self.computation.add_operation(
            op_type(
                placement_name=placement.name,
                name=self.get_fresh_name(f"{op_name}"),
                inputs={"lhs": lhs_operation.name, "rhs": rhs_operation.name},
                signature=OpSignature(
                    input_types={"lhs": lhs_type, "rhs": rhs_type},
                    return_type=expression.vtype,
                ),
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
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=inverse_expression.vtype,
                ),
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
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=abs_expression.vtype,
                ),
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
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=cast_expression.vtype,
                ),
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
                axis=expand_dims_expression.axis,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=expand_dims_expression.vtype,
                ),
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
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=exp_expression.vtype,
                ),
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
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=exp_expression.vtype,
                ),
            )
        )

    def visit_LogExpression(self, log_expression):
        assert isinstance(log_expression, LogExpression)
        (x_expression,) = log_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(log_expression.placement)
        return self.computation.add_operation(
            LogOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("log"),
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=log_expression.vtype,
                ),
            )
        )

    def visit_Log2Expression(self, log2_expression):
        assert isinstance(log2_expression, Log2Expression)
        (x_expression,) = log2_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(log2_expression.placement)
        return self.computation.add_operation(
            Log2Operation(
                placement_name=placement.name,
                name=self.get_fresh_name("log2"),
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=log2_expression.vtype,
                ),
            )
        )

    def visit_SoftmaxExpression(self, softmax_expression):
        assert isinstance(softmax_expression, SoftmaxExpression)
        (x_expression,) = softmax_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(softmax_expression.placement)
        return self.computation.add_operation(
            SoftmaxOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("softmax"),
                axis=softmax_expression.axis,
                upmost_index=softmax_expression.upmost_index,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=softmax_expression.vtype,
                ),
            )
        )

    def visit_ArgmaxExpression(self, argmax_expression):
        assert isinstance(argmax_expression, ArgmaxExpression)
        (x_expression,) = argmax_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(argmax_expression.placement)
        return self.computation.add_operation(
            ArgmaxOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("argmax"),
                axis=argmax_expression.axis,
                upmost_index=argmax_expression.upmost_index,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=argmax_expression.vtype,
                ),
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
                axis=squeeze_expression.axis,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=squeeze_expression.vtype,
                ),
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
                dtype=dtype,
                inputs={"shape": shape_operation.name},
                signature=OpSignature(
                    input_types={"shape": shape_operation.return_type},
                    return_type=ones_expression.vtype,
                ),
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
                axis=sum_expression.axis,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=sum_expression.vtype,
                ),
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
                axis=mean_expression.axis,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=mean_expression.vtype,
                ),
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
                axes=transpose_expression.axes,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=transpose_expression.vtype,
                ),
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
                inputs={"x": x_operation.name, "shape": shape_operation.name},
                signature=OpSignature(
                    input_types={
                        "x": x_operation.return_type,
                        "shape": shape_operation.return_type,
                    },
                    return_type=reshape_expression.vtype,
                ),
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
                to_column_vector=atleast_2d_expression.to_column_vector,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=atleast_2d_expression.vtype,
                ),
            )
        )

    def visit_IndexAxisExpression(self, index_axis_expression):
        assert isinstance(index_axis_expression, IndexAxisExpression)
        (x_expression,) = index_axis_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(index_axis_expression.placement)
        return self.computation.add_operation(
            IndexAxisOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("index_axis"),
                axis=index_axis_expression.axis,
                index=index_axis_expression.index,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=index_axis_expression.vtype,
                ),
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
                begin=slice_expression.begin,
                end=slice_expression.end,
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=slice_expression.vtype,
                ),
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
                inputs={"x": x_operation.name},
                signature=OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=shape_expression.vtype,
                ),
            )
        )

    def visit_MuxExpression(self, mux_expression):
        assert isinstance(mux_expression, MuxExpression)
        (selector_expression, x_expression, y_expression) = mux_expression.inputs
        selector_operation = self.visit(selector_expression)
        x_operation = self.visit(x_expression)
        y_operation = self.visit(y_expression)
        placement = self.visit_placement_expression(mux_expression.placement)
        return self.computation.add_operation(
            MuxOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("mux"),
                inputs={
                    "selector": selector_operation.name,
                    "x": x_operation.name,
                    "y": y_operation.name,
                },
                signature=OpSignature(
                    input_types={
                        "selector": selector_operation.return_type,
                        "x": x_operation.return_type,
                        "y": y_operation.return_type,
                    },
                    return_type=mux_expression.vtype,
                ),
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
                signature=OpSignature(
                    input_types={
                        "key": key_operation.return_type,
                        "query": query_operation.return_type,
                    },
                    return_type=output_type,
                ),
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
                signature=OpSignature(
                    input_types={
                        "key": key_operation.return_type,
                        "value": value_operation.return_type,
                    },
                    return_type=UnitType(),
                ),
            )
        )
