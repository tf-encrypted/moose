import inspect
from collections import defaultdict

from pymoose.computation import computation as comp
from pymoose.computation import operations as ops
from pymoose.computation import placements as plc
from pymoose.computation import types as ty
from pymoose.computation import utils
from pymoose.edsl import base as expr
from pymoose.rust import elk_compiler


def trace(abstract_computation):
    func_signature = inspect.signature(abstract_computation.func)
    symbolic_args = [
        expr.ArgumentExpression(
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
        self.computation = comp.Computation(operations={}, placements={})
        self.name_counters = defaultdict(int)
        self.operation_cache = dict()
        self.placement_cache = dict()

    def trace(self, expressions: expr.Expression) -> comp.Computation:
        if not isinstance(expressions, (tuple, list)):
            expressions = [expressions]
        for expression in expressions:
            output_name = self.get_fresh_name("output")
            op = self.visit(expression)
            self.computation.add_operation(
                ops.OutputOperation(
                    name=output_name,
                    inputs={"value": op.name},
                    placement_name=op.placement_name,
                    signature=ops.OpSignature(
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
        assert isinstance(host_placement_expression, expr.HostPlacementExpression)
        placement = plc.HostPlacement(name=host_placement_expression.name)
        return self.computation.add_placement(placement)

    def visit_ReplicatedPlacementExpression(self, replicated_placement_expression):
        assert isinstance(
            replicated_placement_expression, expr.ReplicatedPlacementExpression
        )
        player_placements = [
            self.visit_placement_expression(player_placement_expression).name
            for player_placement_expression in replicated_placement_expression.players
        ]
        placement = plc.ReplicatedPlacement(
            name=replicated_placement_expression.name, player_names=player_placements
        )
        return self.computation.add_placement(placement)

    def visit_MirroredPlacementExpression(self, mirrored_placement_expression):
        assert isinstance(
            mirrored_placement_expression, expr.MirroredPlacementExpression
        )
        player_placements = [
            self.visit_placement_expression(player_placement_expression).name
            for player_placement_expression in mirrored_placement_expression.players
        ]
        placement = plc.MirroredPlacement(
            name=mirrored_placement_expression.name, player_names=player_placements
        )
        return self.computation.add_placement(placement)

    def visit_IdentityExpression(self, identity_expression):
        assert isinstance(identity_expression, expr.IdentityExpression)
        placement = self.visit_placement_expression(identity_expression.placement)
        input_expression = identity_expression.inputs[0]
        input_op = self.visit(input_expression)
        input_type = input_op.return_type
        output_type = input_expression.vtype
        return self.computation.add_operation(
            ops.IdentityOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("identity"),
                inputs={"x": input_op.name},
                signature=ops.OpSignature(
                    input_types={"x": input_type}, return_type=output_type
                ),
            )
        )

    def visit_ArgumentExpression(self, argument_expression):
        assert isinstance(argument_expression, expr.ArgumentExpression)
        placement = self.visit_placement_expression(argument_expression.placement)
        arg_vtype = argument_expression.vtype
        if arg_vtype is None:
            output_type = ty.UnknownType()
        else:
            output_type = arg_vtype
        return self.computation.add_operation(
            ops.InputOperation(
                placement_name=placement.name,
                name=argument_expression.arg_name,
                inputs={},
                signature=ops.OpSignature(
                    input_types={},
                    return_type=output_type,
                ),
            )
        )

    def visit_AddNExpression(self, add_n_expression):
        assert isinstance(add_n_expression, expr.AddNExpression)
        array_inputs, array_types = {}, {}
        for i, expression in enumerate(add_n_expression.inputs):
            array_op = self.visit(expression)
            array_inputs[f"array{i}"] = array_op.name
            array_types[f"array{i}"] = array_op.return_type

        placement = self.visit_placement_expression(add_n_expression.placement)
        return self.computation.add_operation(
            ops.AddNOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("add_n"),
                inputs=array_inputs,
                signature=ops.OpSignature(
                    input_types=array_types, return_type=add_n_expression.vtype
                ),
            )
        )

    def visit_ConcatenateExpression(self, concatenate_expression):
        assert isinstance(concatenate_expression, expr.ConcatenateExpression)
        array_inputs, array_types = {}, {}
        for i, expression in enumerate(concatenate_expression.inputs):
            array_op = self.visit(expression)
            array_inputs[f"array{i}"] = array_op.name
            array_types[f"array{i}"] = array_op.return_type

        placement = self.visit_placement_expression(concatenate_expression.placement)
        return self.computation.add_operation(
            ops.ConcatenateOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("concatenate"),
                axis=concatenate_expression.axis,
                inputs=array_inputs,
                signature=ops.OpSignature(
                    input_types=array_types,
                    return_type=concatenate_expression.vtype,
                ),
            )
        )

    def visit_MaximumExpression(self, maximum_expression):
        assert isinstance(maximum_expression, expr.MaximumExpression)
        array_inputs, array_types = {}, {}
        for i, expression in enumerate(maximum_expression.inputs):
            array_op = self.visit(expression)
            array_inputs[f"array{i}"] = array_op.name
            array_types[f"array{i}"] = array_op.return_type

        placement = self.visit_placement_expression(maximum_expression.placement)
        return self.computation.add_operation(
            ops.MaximumOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("maximum"),
                inputs=array_inputs,
                signature=ops.OpSignature(
                    input_types=array_types,
                    return_type=maximum_expression.vtype,
                ),
            )
        )

    def visit_DecryptExpression(self, decrypt_expression):
        assert isinstance(decrypt_expression, expr.DecryptExpression)
        assert len(decrypt_expression.inputs) == 2
        aes_key_expression, aes_ciphertext_expression = decrypt_expression.inputs
        aes_key_op = self.visit(aes_key_expression)
        aes_ciphertext_op = self.visit(aes_ciphertext_expression)
        placement = self.visit_placement_expression(decrypt_expression.placement)
        return self.computation.add_operation(
            ops.DecryptOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("decrypt"),
                inputs={"key": aes_key_op.name, "ciphertext": aes_ciphertext_op.name},
                signature=ops.OpSignature(
                    input_types={
                        "key": aes_key_op.return_type,
                        "ciphertext": aes_ciphertext_op.return_type,
                    },
                    return_type=decrypt_expression.vtype,
                ),
            )
        )

    def visit_ConstantExpression(self, constant_expression):
        assert isinstance(constant_expression, expr.ConstantExpression)
        placement = self.visit_placement_expression(constant_expression.placement)
        value = constant_expression.value
        vtype = constant_expression.vtype

        if vtype is None:
            output_type = ty.UnknownType()
        else:
            output_type = vtype
        return self.computation.add_operation(
            ops.ConstantOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("constant"),
                value=value,
                inputs={},
                signature=ops.OpSignature(input_types={}, return_type=output_type),
            )
        )

    def visit_BinaryOpExpression(self, expression):
        assert isinstance(expression, expr.BinaryOpExpression)
        lhs_expression, rhs_expression = expression.inputs
        lhs_operation = self.visit(lhs_expression)
        rhs_operation = self.visit(rhs_expression)
        placement = self.visit_placement_expression(expression.placement)
        op_name = expression.op_name
        op_type = {
            "add": ops.AddOperation,
            "sub": ops.SubOperation,
            "mul": ops.MulOperation,
            "div": ops.DivOperation,
            "dot": ops.DotOperation,
            "or": ops.BitwiseOrOperation,
            "less": ops.LessOperation,
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
                signature=ops.OpSignature(
                    input_types={"lhs": lhs_type, "rhs": rhs_type},
                    return_type=expression.vtype,
                ),
            )
        )

    def visit_InverseExpression(self, inverse_expression):
        assert isinstance(inverse_expression, expr.InverseExpression)
        (x_expression,) = inverse_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(inverse_expression.placement)
        return self.computation.add_operation(
            ops.InverseOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("inverse"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=inverse_expression.vtype,
                ),
            )
        )

    def visit_AbsExpression(self, abs_expression):
        assert isinstance(abs_expression, expr.AbsExpression)
        (x_expression,) = abs_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(abs_expression.placement)
        return self.computation.add_operation(
            ops.AbsOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("abs"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=abs_expression.vtype,
                ),
            )
        )

    def visit_CastExpression(self, cast_expression):
        assert isinstance(cast_expression, expr.CastExpression)
        (x_expression,) = cast_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(cast_expression.placement)
        return self.computation.add_operation(
            ops.CastOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("cast"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=cast_expression.vtype,
                ),
            )
        )

    def visit_ExpandDimsExpression(self, expand_dims_expression):
        assert isinstance(expand_dims_expression, expr.ExpandDimsExpression)
        (x_expression,) = expand_dims_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(expand_dims_expression.placement)
        return self.computation.add_operation(
            ops.ExpandDimsOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("expand_dims"),
                axis=expand_dims_expression.axis,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=expand_dims_expression.vtype,
                ),
            )
        )

    def visit_ExpExpression(self, exp_expression):
        assert isinstance(exp_expression, expr.ExpExpression)
        (x_expression,) = exp_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(exp_expression.placement)
        return self.computation.add_operation(
            ops.ExpOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("exp"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=exp_expression.vtype,
                ),
            )
        )

    def visit_SqrtExpression(self, exp_expression):
        assert isinstance(exp_expression, expr.SqrtExpression)
        (x_expression,) = exp_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(exp_expression.placement)
        return self.computation.add_operation(
            ops.SqrtOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("sqrt"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=exp_expression.vtype,
                ),
            )
        )

    def visit_SigmoidExpression(self, exp_expression):
        assert isinstance(exp_expression, expr.SigmoidExpression)
        (x_expression,) = exp_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(exp_expression.placement)
        return self.computation.add_operation(
            ops.SigmoidOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("sigmoid"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=exp_expression.vtype,
                ),
            )
        )

    def visit_ReluExpression(self, relu_expression):
        assert isinstance(relu_expression, expr.ReluExpression)
        (x_expression,) = relu_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(relu_expression.placement)
        return self.computation.add_operation(
            ops.ReluOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("relu"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=relu_expression.vtype,
                ),
            )
        )

    def visit_LogExpression(self, log_expression):
        assert isinstance(log_expression, expr.LogExpression)
        (x_expression,) = log_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(log_expression.placement)
        return self.computation.add_operation(
            ops.LogOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("log"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=log_expression.vtype,
                ),
            )
        )

    def visit_Log2Expression(self, log2_expression):
        assert isinstance(log2_expression, expr.Log2Expression)
        (x_expression,) = log2_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(log2_expression.placement)
        return self.computation.add_operation(
            ops.Log2Operation(
                placement_name=placement.name,
                name=self.get_fresh_name("log2"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=log2_expression.vtype,
                ),
            )
        )

    def visit_SoftmaxExpression(self, softmax_expression):
        assert isinstance(softmax_expression, expr.SoftmaxExpression)
        (x_expression,) = softmax_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(softmax_expression.placement)
        return self.computation.add_operation(
            ops.SoftmaxOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("softmax"),
                axis=softmax_expression.axis,
                upmost_index=softmax_expression.upmost_index,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=softmax_expression.vtype,
                ),
            )
        )

    def visit_ArgmaxExpression(self, argmax_expression):
        assert isinstance(argmax_expression, expr.ArgmaxExpression)
        (x_expression,) = argmax_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(argmax_expression.placement)
        return self.computation.add_operation(
            ops.ArgmaxOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("argmax"),
                axis=argmax_expression.axis,
                upmost_index=argmax_expression.upmost_index,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=argmax_expression.vtype,
                ),
            )
        )

    def visit_SqueezeExpression(self, squeeze_expression):
        assert isinstance(squeeze_expression, expr.SqueezeExpression)
        (x_expression,) = squeeze_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(squeeze_expression.placement)
        return self.computation.add_operation(
            ops.SqueezeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("squeeze"),
                axis=squeeze_expression.axis,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=squeeze_expression.vtype,
                ),
            )
        )

    def visit_OnesExpression(self, ones_expression):
        assert isinstance(ones_expression, expr.OnesExpression)
        (shape_expression,) = ones_expression.inputs
        shape_operation = self.visit(shape_expression)
        placement = self.visit_placement_expression(ones_expression.placement)
        return self.computation.add_operation(
            ops.OnesOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("ones"),
                inputs={"shape": shape_operation.name},
                signature=ops.OpSignature(
                    input_types={"shape": shape_operation.return_type},
                    return_type=ones_expression.vtype,
                ),
            )
        )

    def visit_ZerosExpression(self, zeros_expression):
        assert isinstance(zeros_expression, expr.ZerosExpression)
        (shape_expression,) = zeros_expression.inputs
        shape_operation = self.visit(shape_expression)
        placement = self.visit_placement_expression(zeros_expression.placement)
        return self.computation.add_operation(
            ops.ZerosOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("zeros"),
                inputs={"shape": shape_operation.name},
                signature=ops.OpSignature(
                    input_types={"shape": shape_operation.return_type},
                    return_type=zeros_expression.vtype,
                ),
            )
        )

    def visit_SumExpression(self, sum_expression):
        assert isinstance(sum_expression, expr.SumExpression)
        (x_expression,) = sum_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(sum_expression.placement)
        return self.computation.add_operation(
            ops.SumOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("sum"),
                axis=sum_expression.axis,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=sum_expression.vtype,
                ),
            )
        )

    def visit_MeanExpression(self, mean_expression):
        assert isinstance(mean_expression, expr.MeanExpression)
        (x_expression,) = mean_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(mean_expression.placement)
        return self.computation.add_operation(
            ops.MeanOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("mean"),
                axis=mean_expression.axis,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=mean_expression.vtype,
                ),
            )
        )

    def visit_TransposeExpression(self, transpose_expression):
        assert isinstance(transpose_expression, expr.TransposeExpression)
        (x_expression,) = transpose_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(transpose_expression.placement)
        return self.computation.add_operation(
            ops.TransposeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("transpose"),
                axes=transpose_expression.axes,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=transpose_expression.vtype,
                ),
            )
        )

    def visit_ReshapeExpression(self, reshape_expression):
        assert isinstance(reshape_expression, expr.ReshapeExpression)
        (x_expression, shape_expression) = reshape_expression.inputs
        x_operation = self.visit(x_expression)
        shape_operation = self.visit(shape_expression)
        placement = self.visit_placement_expression(reshape_expression.placement)
        return self.computation.add_operation(
            ops.ReshapeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("reshape"),
                inputs={"x": x_operation.name, "shape": shape_operation.name},
                signature=ops.OpSignature(
                    input_types={
                        "x": x_operation.return_type,
                        "shape": shape_operation.return_type,
                    },
                    return_type=reshape_expression.vtype,
                ),
            )
        )

    def visit_AtLeast2DExpression(self, atleast_2d_expression):
        assert isinstance(atleast_2d_expression, expr.AtLeast2DExpression)
        (x_expression,) = atleast_2d_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(atleast_2d_expression.placement)
        return self.computation.add_operation(
            ops.AtLeast2DOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("atleast_2d"),
                to_column_vector=atleast_2d_expression.to_column_vector,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=atleast_2d_expression.vtype,
                ),
            )
        )

    def visit_IndexAxisExpression(self, index_axis_expression):
        assert isinstance(index_axis_expression, expr.IndexAxisExpression)
        (x_expression,) = index_axis_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(index_axis_expression.placement)
        return self.computation.add_operation(
            ops.IndexAxisOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("index_axis"),
                axis=index_axis_expression.axis,
                index=index_axis_expression.index,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=index_axis_expression.vtype,
                ),
            )
        )

    def visit_SliceExpression(self, slice_expression):
        assert isinstance(slice_expression, expr.SliceExpression)
        (x_expression,) = slice_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(slice_expression.placement)
        return self.computation.add_operation(
            ops.SliceOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("slice"),
                begin=slice_expression.begin,
                end=slice_expression.end,
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=slice_expression.vtype,
                ),
            )
        )

    def visit_ShapeExpression(self, shape_expression):
        assert isinstance(shape_expression, expr.ShapeExpression)
        (x_expression,) = shape_expression.inputs
        x_operation = self.visit(x_expression)
        placement = self.visit_placement_expression(shape_expression.placement)
        return self.computation.add_operation(
            ops.ShapeOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("shape"),
                inputs={"x": x_operation.name},
                signature=ops.OpSignature(
                    input_types={"x": x_operation.return_type},
                    return_type=shape_expression.vtype,
                ),
            )
        )

    def visit_MuxExpression(self, mux_expression):
        assert isinstance(mux_expression, expr.MuxExpression)
        (selector_expression, x_expression, y_expression) = mux_expression.inputs
        selector_operation = self.visit(selector_expression)
        x_operation = self.visit(x_expression)
        y_operation = self.visit(y_expression)
        placement = self.visit_placement_expression(mux_expression.placement)
        return self.computation.add_operation(
            ops.MuxOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("mux"),
                inputs={
                    "selector": selector_operation.name,
                    "x": x_operation.name,
                    "y": y_operation.name,
                },
                signature=ops.OpSignature(
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
        assert isinstance(load_expression, expr.LoadExpression)
        key_expression, query_expression = load_expression.inputs
        key_operation = self.visit(key_expression)
        query_operation = self.visit(query_expression)
        placement = self.visit_placement_expression(load_expression.placement)
        output_type = load_expression.vtype or ty.UnknownType()
        return self.computation.add_operation(
            ops.LoadOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("load"),
                inputs={"key": key_operation.name, "query": query_operation.name},
                signature=ops.OpSignature(
                    input_types={
                        "key": key_operation.return_type,
                        "query": query_operation.return_type,
                    },
                    return_type=output_type,
                ),
            )
        )

    def visit_SaveExpression(self, save_expression):
        assert isinstance(save_expression, expr.SaveExpression)
        (key_expression, value_expression) = save_expression.inputs
        key_operation = self.visit(key_expression)
        value_operation = self.visit(value_expression)
        placement = self.visit_placement_expression(save_expression.placement)
        return self.computation.add_operation(
            ops.SaveOperation(
                placement_name=placement.name,
                name=self.get_fresh_name("save"),
                inputs={"key": key_operation.name, "value": value_operation.name},
                signature=ops.OpSignature(
                    input_types={
                        "key": key_operation.return_type,
                        "value": value_operation.return_type,
                    },
                    return_type=ty.UnitType(),
                ),
            )
        )
