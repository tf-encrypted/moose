from moose.compiler.pruning import PruningPass
from moose.computation import fixedpoint as fixedpoint_dialect
from moose.computation import standard as std_dialect


class HostEncodingPass:
    def __init__(self):
        self.computation = None
        self.context = None

    def run(self, computation, context):
        self.computation = computation
        self.context = context

        ops_to_replace = []
        for op in self.computation.operations.values():
            if not isinstance(op, std_dialect.CastOperation):
                continue
            [(input_key, input_op_name)] = op.inputs.items()
            input_op = self.computation.operation(input_op_name)
            assert hasattr(input_op.output_type, "dtype"), input_op.output_type
            input_dtype = input_op.output_type.dtype
            output_dtype = op.output_type.dtype
            assert not input_dtype.is_fixedpoint or not output_dtype.is_fixedpoint
            if input_dtype.is_fixedpoint or output_dtype.is_fixedpoint:
                ops_to_replace.append((op, input_dtype, output_dtype))

        for op, input_dtype, output_dtype in ops_to_replace:
            if output_dtype.is_fixedpoint:
                self.lower_to_encode(op)
                performed_changes = True
            elif input_dtype.is_fixedpoint:
                self.lower_to_decode(op)
                performed_changes = True
            else:
                raise ValueError("HostEncodingPass encountered improper CastOperation.")

        # prune old ops
        pruning_pass = PruningPass()
        computation, pruning_performed_changes = pruning_pass.run(computation, context)
        performed_changes = len(ops_to_replace) > 0 or pruning_performed_changes

        return computation, performed_changes

    def lower_to_encode(self, op):
        assert isinstance(op, std_dialect.CastOperation)
        assert op.output_type.dtype.is_fixedpoint
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
        self._rewire_output_ops(op, encode_op)

    def lower_to_decode(self, op):
        assert isinstance(op, std_dialect.CastOperation)
        assert len(op.inputs) == 1
        assert op.output_type.dtype.is_float
        [(input_key, input_op_name)] = op.inputs.items()
        input_op = self.computation.operation(input_op_name)
        # breakpoint()
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
        self._rewire_output_ops(op, decode_op)

    def _rewire_output_ops(self, old_src_op, new_src_op):
        dst_ops = self.computation.find_destinations(old_src_op)
        for dst_op in dst_ops:
            updated_wirings = {
                k: new_src_op.name
                for k, v in dst_op.inputs.items()
                if v == old_src_op.name
            }
            dst_op.inputs.update(updated_wirings)
