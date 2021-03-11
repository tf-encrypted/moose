from moose.compiler import substitution_pass
from moose.computation import fixedpoint as fixedpoint_dialect
from moose.computation import standard as std_dialect


class HostEncodingPass(substitution_pass.SubstitutionPass):
    """Convert casting ops with fixedpoint dtypes to encode/decode ops."""

    def qualify_substitution(self, op):
        if not isinstance(op, std_dialect.CastOperation):
            return False
        [(input_key, input_op_name)] = op.inputs.items()
        input_op = self.computation.operation(input_op_name)
        assert hasattr(input_op.output_type, "dtype"), input_op.output_type
        input_dtype = input_op.output_type.dtype
        output_dtype = op.output_type.dtype
        assert not input_dtype.is_fixedpoint or not output_dtype.is_fixedpoint
        if input_dtype.is_fixedpoint or output_dtype.is_fixedpoint:
            return True
        return False

    def lower(self, op_name):
        op = self.computation.operation(op_name)
        if op.output_type.dtype.is_fixedpoint:
            # cast output is fixedpoint, so encode
            lowering_fn = self.lower_to_encode
        else:
            # cast input is fixedpoint, so decode
            lowering_fn = self.lower_to_decode
        lowered_op = lowering_fn(op)
        return lowered_op

    def lower_to_encode(self, op):
        assert isinstance(op, std_dialect.CastOperation)
        assert len(op.inputs) == 1
        assert op.output_type.dtype.is_fixedpoint
        [(input_key, input_op_name)] = op.inputs.items()
        input_op = self.computation.operation(input_op_name)
        assert input_op.output_type.dtype.is_float
        precision = op.output_type.dtype.fractional_precision
        encode_op = self.computation.add_operation(
            fixedpoint_dialect.EncodeOperation(
                name=self.context.get_fresh_name("encode"),
                placement_name=op.placement_name,
                inputs=op.inputs,
                output_type=fixedpoint_dialect.EncodedTensorType(
                    dtype=op.output_type.dtype, precision=precision,
                ),
                precision=precision,
            )
        )
        return encode_op

    def lower_to_decode(self, op):
        assert isinstance(op, std_dialect.CastOperation)
        assert len(op.inputs) == 1
        assert op.output_type.dtype.is_float
        [(input_key, input_op_name)] = op.inputs.items()
        input_op = self.computation.operation(input_op_name)
        assert input_op.output_type.dtype.is_fixedpoint
        precision = input_op.output_type.dtype.fractional_precision
        decode_op = self.computation.add_operation(
            fixedpoint_dialect.DecodeOperation(
                name=self.context.get_fresh_name("encode"),
                placement_name=op.placement_name,
                inputs=op.inputs,
                output_type=op.output_type,
                # TODO make this more than just float64 output dtype
                precision=precision,
            )
        )
        return decode_op
