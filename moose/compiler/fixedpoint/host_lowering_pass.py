from moose.compiler import host as host_dialect
from moose.compiler import substitution_pass
from moose.computation import dtypes
from moose.computation import fixedpoint as fixedpoint_dialect
from moose.computation import standard as std_dialect


class HostLoweringPass(substitution_pass.SubstitutionPass):
    """Lower standard ops with fixedpoint dtype into fixedpoint ops."""

    def qualify_substitution(self, op):
        if not isinstance(op, std_dialect.StandardOperation):
            return False
        op_placement = self.computation.placement(op.placement_name)
        if not isinstance(op_placement, host_dialect.HostPlacement):
            return False
        if hasattr(op.output_type, "dtype") and op.output_type.dtype.is_fixedpoint:
            return True
        return False

    def lower(self, op_name):
        op = self.computation.operation(op_name)
        # lower op based on type
        lowering_fn = getattr(self, f"lower_{type(op).__name__}", None)
        if lowering_fn is None:
            raise NotImplementedError(f"{type(op)}")
        lowered_op = lowering_fn(op)
        return lowered_op

    def lower_MulOperation(self, op):
        assert isinstance(op, std_dialect.MulOperation)
        assert len(op.inputs) == 2
        assert op.output_type.dtype.is_fixedpoint
        op_dtype = op.output_type.dtype
        input_ops = [
            self.computation.operation(input_op_name)
            for _, input_op_name in op.inputs.items()
        ]
        assert all(inp.output_type.dtype.is_fixedpoint for inp in input_ops)
        mul_precision = sum(
            inp.output_type.dtype.fractional_precision for inp in input_ops
        )
        # TODO(jason): double-check integral precision here
        mul_dtype = dtypes.fixed(op_dtype.integral_precision, mul_precision)
        mul_op = self.computation.add_operation(
            fixedpoint_dialect.MulOperation(
                name=self.context.get_fresh_name("fixed_mul"),
                placement_name=op.placement_name,
                inputs=op.inputs,
                output_type=fixedpoint_dialect.EncodedTensorType(
                    dtype=mul_dtype, precision=mul_precision,
                ),
            )
        )
        trunc_output_type = fixedpoint_dialect.EncodedTensorType(
            dtype=op_dtype, precision=mul_precision // 2
        )
        trunc_op = self.computation.add_operation(
            fixedpoint_dialect.TruncOperation(
                name=self.context.get_fresh_name("fixed_trunc"),
                placement_name=op.placement_name,
                inputs={"value": mul_op.name},
                output_type=trunc_output_type,
                precision=mul_precision - trunc_output_type.precision,
            )
        )
        return trunc_op
