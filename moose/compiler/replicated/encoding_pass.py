from moose.compiler.replicated.subgraph_replace_pass import SubgraphReplacementPass
from moose.computation import fixedpoint as fixedpoint_dialect
from moose.computation import replicated as rep_dialect
from moose.computation import standard as std_dialect


class ReplicatedEncodingPass(SubgraphReplacementPass):
    """Lower standard ops to fixedpoint ops on replicated placements.
    """

    def __init__(self):
        super().__init__()

    def collect_subgraph(self):
        op_names_to_process = set()
        for op in self.computation.operations.values():
            if not isinstance(op, std_dialect.StandardOperation):
                continue
            op_placement = self.computation.placement(op.placement_name)
            if not isinstance(op_placement, rep_dialect.ReplicatedPlacement):
                continue
            op_names_to_process.add(op.name)
        return op_names_to_process

    def process_AddOperation(self, op, processed_inputs):
        assert isinstance(op, std_dialect.AddOperation)
        lowered_lhs_op = processed_inputs["lhs"]
        lowered_rhs_op = processed_inputs["rhs"]
        lhs_output_type = lowered_lhs_op.output_type
        rhs_output_type = lowered_rhs_op.output_type
        assert isinstance(lhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert isinstance(rhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert lhs_output_type.dtype == rhs_output_type.dtype
        assert lhs_output_type.precision == rhs_output_type.precision
        output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=lhs_output_type.dtype, precision=lhs_output_type.precision,
        )
        return self.computation.add(
            fixedpoint_dialect.AddOperation(
                name=self.context.get_fresh_name("fixed_add"),
                placement_name=op.placement_name,
                inputs={"lhs": lowered_lhs_op.name, "rhs": lowered_rhs_op.name},
                output_type=output_type,
            )
        )

    def process_SubOperation(self, op, processed_inputs):
        assert isinstance(op, std_dialect.SubOperation)
        lowered_lhs_op = processed_inputs["lhs"]
        lowered_rhs_op = processed_inputs["rhs"]
        lhs_output_type = lowered_lhs_op.output_type
        rhs_output_type = lowered_rhs_op.output_type
        assert isinstance(lhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert isinstance(rhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert lhs_output_type.dtype == rhs_output_type.dtype
        assert lhs_output_type.precision == rhs_output_type.precision
        output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=lhs_output_type.dtype, precision=lhs_output_type.precision,
        )
        return self.computation.add(
            fixedpoint_dialect.SubOperation(
                name=self.context.get_fresh_name("fixed_add"),
                placement_name=op.placement_name,
                inputs={"lhs": lowered_lhs_op.name, "rhs": lowered_rhs_op.name},
                output_type=output_type,
            )
        )

    def process_MulOperation(self, op, processed_inputs):
        assert isinstance(op, std_dialect.MulOperation)
        lowered_lhs_op = processed_inputs["lhs"]
        lowered_rhs_op = processed_inputs["rhs"]
        lhs_output_type = lowered_lhs_op.output_type
        rhs_output_type = lowered_rhs_op.output_type
        assert isinstance(lhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert isinstance(rhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert lhs_output_type.dtype == rhs_output_type.dtype
        assert lhs_output_type.precision == rhs_output_type.precision
        mul_output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=lhs_output_type.dtype,
            precision=lhs_output_type.precision + rhs_output_type.precision,
        )
        mul_op = self.computation.add(
            fixedpoint_dialect.MulOperation(
                name=self.context.get_fresh_name("fixed_mul"),
                placement_name=op.placement_name,
                inputs={"lhs": lowered_lhs_op.name, "rhs": lowered_rhs_op.name},
                output_type=mul_output_type,
            )
        )
        if lhs_output_type.precision == 0 or rhs_output_type.precision == 0:
            return mul_op

        trunc_output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=mul_output_type.dtype, precision=mul_output_type.precision // 2,
        )
        precision_to_truncate = mul_output_type.precision - trunc_output_type.precision
        trunc_op = self.computation.add(
            fixedpoint_dialect.TruncPrOperation(
                name=self.context.get_fresh_name("trunc_pr"),
                placement_name=op.placement_name,
                inputs={"value": mul_op.name},
                precision=precision_to_truncate,
                output_type=trunc_output_type,
            )
        )
        return trunc_op

    def process_DotOperation(self, op, processed_inputs):
        assert isinstance(op, std_dialect.DotOperation)
        lowered_lhs_op = processed_inputs["lhs"]
        lowered_rhs_op = processed_inputs["rhs"]
        lhs_output_type = lowered_lhs_op.output_type
        rhs_output_type = lowered_rhs_op.output_type
        assert isinstance(lhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert isinstance(rhs_output_type, fixedpoint_dialect.EncodedTensorType)
        assert lhs_output_type.dtype == rhs_output_type.dtype
        assert lhs_output_type.precision == rhs_output_type.precision
        dot_output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=lhs_output_type.dtype,
            precision=lhs_output_type.precision + rhs_output_type.precision,
        )
        dot_op = self.computation.add(
            fixedpoint_dialect.DotOperation(
                name=self.context.get_fresh_name("fixed_dot"),
                placement_name=op.placement_name,
                inputs={"lhs": lowered_lhs_op.name, "rhs": lowered_rhs_op.name},
                output_type=dot_output_type,
            )
        )
        if lhs_output_type.precision == 0 or rhs_output_type.precision == 0:
            return dot_op

        trunc_output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=dot_output_type.dtype, precision=dot_output_type.precision // 2,
        )
        precision_to_truncate = dot_output_type.precision - trunc_output_type.precision
        trunc_op = self.computation.add(
            fixedpoint_dialect.TruncPrOperation(
                name=self.context.get_fresh_name("trunc_pr"),
                placement_name=op.placement_name,
                inputs={"value": dot_op.name},
                precision=precision_to_truncate,
                output_type=trunc_output_type,
            )
        )
        return trunc_op

    def process_SumOperation(self, op, processed_inputs):
        assert isinstance(op, std_dialect.SumOperation)
        lowered_x_op = processed_inputs["x"]
        x_output_type = lowered_x_op.output_type
        assert isinstance(x_output_type, fixedpoint_dialect.EncodedTensorType)
        sum_op = self.computation.add(
            fixedpoint_dialect.SumOperation(
                name=self.context.get_fresh_name("fixed_sum"),
                placement_name=op.placement_name,
                axis=op.axis,
                inputs={"x": lowered_x_op.name},
                output_type=x_output_type,
            )
        )
        return sum_op

    def process_MeanOperation(self, op, processed_inputs):
        assert isinstance(op, std_dialect.MeanOperation)
        lowered_arg_op = processed_inputs["x"]
        arg_output_type = lowered_arg_op.output_type
        assert isinstance(arg_output_type, fixedpoint_dialect.EncodedTensorType)
        mean_output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=arg_output_type.dtype, precision=2 * arg_output_type.precision,
        )
        mean_op = self.computation.add(
            fixedpoint_dialect.MeanOperation(
                name=self.context.get_fresh_name("fixed_mean"),
                placement_name=op.placement_name,
                inputs={"x": lowered_arg_op.name},
                axis=op.axis,
                precision=arg_output_type.precision,
                output_type=mean_output_type,
                scaling_base=2,
                scaling_exp=arg_output_type.precision
            )
        )
        if mean_output_type.precision == 0:
            return mean_op

        trunc_output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=mean_output_type.dtype, precision=mean_output_type.precision // 2,
        )
        precision_to_truncate = mean_output_type.precision - trunc_output_type.precision
        trunc_op = self.computation.add(
            fixedpoint_dialect.TruncPrOperation(
                name=self.context.get_fresh_name("trunc_pr"),
                placement_name=op.placement_name,
                inputs={"value": mean_op.name},
                precision=precision_to_truncate,
                output_type=trunc_output_type,
            )
        )
        return trunc_op

    def process_AbsOperation(self, op, processed_inputs):
        assert isinstance(op, std_dialect.AbsOperation)
        lowered_x_op = processed_inputs["x"]
        x_output_type = lowered_x_op.output_type
        assert isinstance(x_output_type, fixedpoint_dialect.EncodedTensorType)
        abs_op = self.computation.add(
            fixedpoint_dialect.AbsOperation(
                name=self.context.get_fresh_name("fixed_abs"),
                placement_name=op.placement_name,
                inputs={"x": lowered_x_op.name},
                output_type=x_output_type,
            )
        )
        return abs_op

    def process_incoming_edge(self, src_op_name, input_key, dst_op_name):
        src_op = self.computation.operation(src_op_name)
        assert isinstance(src_op, fixedpoint_dialect.FixedpointOperation)
        return src_op

    def process_outgoing_edge(self, src_op, input_key, dst_op_name):
        assert isinstance(src_op, fixedpoint_dialect.FixedpointOperation), type(src_op)
        return src_op.name
