from moose.compiler import host as host_dialect
from moose.compiler.pruning import PruningPass
from moose.computation import dtypes
from moose.computation import fixedpoint as fixedpoint_dialect
from moose.computation import standard as std_dialect


class HostLoweringPass:
    """Lower standard ops with fixedpoint dtype into fixedpoint ops."""

    def __init__(self):
        self.computation = None
        self.context = None

    def run(self, computation, context):
        self.computation = computation
        self.context = context

        # determine which ops we should lower; in this case, it's
        # ops that have a fixedpoint output dtype
        op_names_to_lower = set()
        for op in computation.operations.values():
            if not isinstance(op, std_dialect.StandardOperation):
                continue
            op_placement = computation.placement(op.placement_name)
            if not isinstance(op_placement, host_dialect.HostPlacement):
                continue
            if hasattr(op.output_type, "dtype") and op.output_type.dtype.is_fixedpoint:
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
