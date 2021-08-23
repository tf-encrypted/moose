from pymoose.computation import ring as ring_dialect
from pymoose.deprecated.compiler import host as host_dialect
from pymoose.deprecated.compiler import substitution_pass
from pymoose.deprecated.computation import fixedpoint as fixedpoint_dialect


class HostRingLoweringPass(substitution_pass.SubstitutionPass):
    """Lower fixedpoint ops into ring ops on HostPlacement."""

    def qualify_substitution(self, op):
        if not isinstance(op, fixedpoint_dialect.FixedpointOperation):
            return False
        op_placement = self.computation.placement(op.placement_name)
        if not isinstance(op_placement, host_dialect.HostPlacement):
            return False
        return True

    def lower(self, op_name):
        op = self.computation.operation(op_name)
        # lower op based on type
        lowering_fn = getattr(self, f"lower_{type(op).__name__}", None)
        if lowering_fn is None:
            raise NotImplementedError(f"{type(op)}")
        lowered_op = lowering_fn(op)
        return lowered_op

    def lower_EncodeOperation(self, op):
        assert isinstance(op, fixedpoint_dialect.EncodeOperation)
        assert len(op.inputs) == 1
        [(input_key, input_op_name)] = op.inputs.items()
        input_op = self.computation.operation(input_op_name)
        assert input_op.output_type.dtype.is_float
        assert isinstance(op.output_type, fixedpoint_dialect.EncodedTensorType)
        return self.computation.add_operation(
            fixedpoint_dialect.RingEncodeOperation(
                name=self.context.get_fresh_name("ring_encode"),
                placement_name=op.placement_name,
                inputs={"value": input_op_name},
                scaling_base=2,
                scaling_exp=op.precision,
            )
        )

    def lower_DecodeOperation(self, op):
        assert isinstance(op, fixedpoint_dialect.DecodeOperation)
        assert len(op.inputs) == 1
        [(input_key, input_op_name)] = op.inputs.items()
        input_op = self.computation.operation(input_op_name)
        assert input_op.output_type.dtype.is_fixedpoint
        assert op.output_type.dtype.is_float
        return self.computation.add_operation(
            fixedpoint_dialect.RingDecodeOperation(
                name=self.context.get_fresh_name("ring_decode"),
                placement_name=op.placement_name,
                inputs={"value": input_op_name},
                output_type=op.output_type,
                scaling_base=2,
                scaling_exp=op.precision,
            )
        )

    def lower_AddOperation(self, op):
        assert isinstance(op, fixedpoint_dialect.AddOperation)
        assert len(op.inputs) == 2
        assert isinstance(op.output_type, fixedpoint_dialect.EncodedTensorType)
        return self.computation.add_operation(
            ring_dialect.RingAddOperation(
                name=self.context.get_fresh_name("ring_add"),
                placement_name=op.placement_name,
                inputs=op.inputs,
            )
        )

    def lower_MulOperation(self, op):
        assert isinstance(op, fixedpoint_dialect.MulOperation)
        assert len(op.inputs) == 2
        assert isinstance(op.output_type, fixedpoint_dialect.EncodedTensorType)
        return self.computation.add_operation(
            ring_dialect.RingMulOperation(
                name=self.context.get_fresh_name("ring_mul"),
                placement_name=op.placement_name,
                inputs=op.inputs,
            )
        )

    def lower_SubOperation(self, op):
        assert isinstance(op, fixedpoint_dialect.SubOperation)
        assert len(op.inputs) == 2
        assert isinstance(op.output_type, fixedpoint_dialect.EncodedTensorType)
        return self.computation.add_operation(
            ring_dialect.RingSubOperation(
                name=self.context.get_fresh_name("ring_sub"),
                placement_name=op.placement_name,
                inputs=op.inputs,
            )
        )

    def lower_SumOperation(self, op):
        assert isinstance(op, fixedpoint_dialect.SumOperation)
        assert len(op.inputs) == 1
        assert isinstance(op.output_type, fixedpoint_dialect.EncodedTensorType)
        return self.computation.add_operation(
            ring_dialect.RingSumOperation(
                name=self.context.get_fresh_name("ring_sum"),
                placement_name=op.placement_name,
                inputs=op.inputs,
                axis=op.axis,
            )
        )

    def lower_TruncOperation(self, op):
        assert isinstance(op, fixedpoint_dialect.TruncOperation)
        assert len(op.inputs) == 1
        assert isinstance(op.output_type, fixedpoint_dialect.EncodedTensorType)
        amount = op.precision
        assert amount < 64
        return self.computation.add_operation(
            ring_dialect.RingShrOperation(
                name=self.context.get_fresh_name("ring_shr"),
                placement_name=op.placement_name,
                inputs=op.inputs,
                amount=amount,
            )
        )
