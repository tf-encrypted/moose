from moose.compiler import host as host_dialect
from moose.compiler.pruning import PruningPass
from moose.computation import fixedpoint as fixedpoint_dialect
from moose.computation import ring as ring_dialect


class RingLoweringPass:
    """Lower fixedpoint ops into ring ops."""

    def __init__(self):
        self.computation = None
        self.context = None

    def run(self, computation, context):
        self.computation = computation
        self.context = context

        # determine which ops we should lower; these should just be
        # any fixedpoint ops found on host placements.
        op_names_to_lower = set()
        for op in computation.operations.values():
            if not isinstance(op, fixedpoint_dialect.FixedpointOperation):
                continue
            op_placement = computation.placement(op.placement_name)
            if not isinstance(op_placement, host_dialect.HostPlacement):
                continue
            op_names_to_lower.add(op.name)
        # lower the ops
        op_names_to_rewire = set()
        for op_name in op_names_to_lower:
            lowered_op = self.lower(op_name)
            op_names_to_rewire.add((lowered_op.name, op_name))
        # rewire outputs of lowered ops
        for lowered_op_name, old_op_name in op_names_to_rewire:
            old_op = computation.operation(old_op_name)
            lowered_op = computation.operation(lowered_op_name)
            self._rewire_output_ops(old_op, lowered_op)
        # prune old ops
        pruning_pass = PruningPass()
        computation, pruning_performed_changes = pruning_pass.run(computation, context)
        # if we changed the graph at all, let the compiler know
        performed_changes = len(op_names_to_lower) > 0 or pruning_performed_changes
        return computation, performed_changes

    def lower(self, op_name):
        op = self.computation.operation(op_name)
        # lower op based on type
        lowering_fn = getattr(self, f"lower_{type(op).__name__}", None)
        if lowering_fn is None:
            raise NotImplementedError(f"{type(op)}")
        lowered_op = lowering_fn(op)
        return lowered_op

    def _rewire_output_ops(self, old_src_op, new_src_op):
        dst_ops = self.computation.find_destinations(old_src_op)
        for dst_op in dst_ops:
            updated_wirings = {
                k: new_src_op.name
                for k, v in dst_op.inputs.items()
                if v == old_src_op.name
            }
            dst_op.inputs.update(updated_wirings)

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
                scaling_factor=2 ** op.precision,
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
                scaling_factor=2 ** op.precision,
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
