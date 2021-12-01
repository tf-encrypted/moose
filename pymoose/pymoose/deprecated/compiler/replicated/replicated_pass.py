from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.deprecated.compiler.replicated import subgraph_replace_pass
from pymoose.deprecated.computation import fixedpoint as fixed_dialect
from pymoose.deprecated.computation import replicated as rep_dialect


class ReplicatedOpsPass(subgraph_replace_pass.SubgraphReplacementPass):
    """Lower fixedpoint ops to replicated ops on replicated placements.
    """

    def __init__(self):
        super().__init__()
        self.incoming_edge_cache = None
        self.outgoing_edge_cache = None
        self.setup_cache = None

    def run(self, computation, context):
        self.incoming_edge_cache = dict()
        self.outgoing_edge_cache = dict()
        self.setup_cache = dict()
        return super().run(computation, context)

    def collect_subgraph(self):
        op_names_to_process = set()
        for op in self.computation.operations.values():
            op_placement = self.computation.placement(op.placement_name)
            if not isinstance(op_placement, ReplicatedPlacement):
                continue
            if not isinstance(op, fixed_dialect.FixedpointOperation):
                continue
            op_names_to_process.add(op.name)
        return op_names_to_process

    def get_setup_op(self, replicated_placement_name):
        cache_key = replicated_placement_name
        if cache_key not in self.setup_cache:
            self.setup_cache[cache_key] = self.computation.add_operation(
                rep_dialect.SetupOperation(
                    name=self.context.get_fresh_name("replicated_setup"),
                    placement_name=replicated_placement_name,
                    inputs={},
                )
            )
        return self.setup_cache[cache_key]

    def process_AddOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.AddOperation)
        inputs = {
            input_key: input_op.name for input_key, input_op in processed_inputs.items()
        }
        assert "setup" not in inputs
        inputs["setup"] = self.get_setup_op(op.placement_name).name
        return self.computation.add_operation(
            rep_dialect.AddOperation(
                name=self.context.get_fresh_name("replicated_add"),
                placement_name=op.placement_name,
                inputs=inputs,
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_SubOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.SubOperation)
        inputs = {
            input_key: input_op.name for input_key, input_op in processed_inputs.items()
        }
        assert "setup" not in inputs
        inputs["setup"] = self.get_setup_op(op.placement_name).name
        return self.computation.add_operation(
            rep_dialect.SubOperation(
                name=self.context.get_fresh_name("replicated_sub"),
                placement_name=op.placement_name,
                inputs=inputs,
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_MulOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.MulOperation)
        inputs = {
            input_key: input_op.name for input_key, input_op in processed_inputs.items()
        }
        assert "setup" not in inputs
        inputs["setup"] = self.get_setup_op(op.placement_name).name
        return self.computation.add_operation(
            rep_dialect.MulOperation(
                name=self.context.get_fresh_name("replicated_mul"),
                placement_name=op.placement_name,
                inputs=inputs,
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_TruncPrOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.TruncPrOperation)
        inputs = {
            input_key: input_op.name for input_key, input_op in processed_inputs.items()
        }
        assert "setup" not in inputs
        inputs["setup"] = self.get_setup_op(op.placement_name).name
        return self.computation.add_operation(
            rep_dialect.TruncPrOperation(
                name=self.context.get_fresh_name("replicated_trunc_pr"),
                placement_name=op.placement_name,
                inputs=inputs,
                precision=op.precision,
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_DotOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.DotOperation)
        inputs = {
            input_key: input_op.name for input_key, input_op in processed_inputs.items()
        }
        assert "setup" not in inputs
        inputs["setup"] = self.get_setup_op(op.placement_name).name
        return self.computation.add_operation(
            rep_dialect.DotOperation(
                name=self.context.get_fresh_name("replicated_dot"),
                placement_name=op.placement_name,
                inputs=inputs,
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_AbsOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.AbsOperation)
        inputs = {
            input_key: input_op.name for input_key, input_op in processed_inputs.items()
        }
        assert "setup" not in inputs
        inputs["setup"] = self.get_setup_op(op.placement_name).name
        return self.computation.add_operation(
            rep_dialect.AbsOperation(
                name=self.context.get_fresh_name("replicated_abs"),
                placement_name=op.placement_name,
                inputs=inputs,
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_SumOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.SumOperation)
        inputs = {
            input_key: input_op.name for input_key, input_op in processed_inputs.items()
        }
        return self.computation.add_operation(
            rep_dialect.SumOperation(
                name=self.context.get_fresh_name("replicated_sum"),
                placement_name=op.placement_name,
                axis=op.axis,
                inputs=inputs,
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_MeanOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.MeanOperation)
        return self.computation.add_operation(
            rep_dialect.MeanOperation(
                name=self.context.get_fresh_name("mean"),
                placement_name=op.placement_name,
                axis=op.axis,
                precision=op.precision,
                inputs={
                    input_key: input_op.name
                    for input_key, input_op in processed_inputs.items()
                },
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=op.output_type.dtype
                ),
            )
        )

    def process_incoming_edge(self, src_op_name, input_key, dst_op_name):
        src_op = self.computation.operation(src_op_name)
        dst_op = self.computation.operation(dst_op_name)
        cache_key = (src_op.name, dst_op.placement_name)
        if cache_key not in self.incoming_edge_cache:
            share_op = self.computation.add_operation(
                rep_dialect.ShareOperation(
                    name=self.context.get_fresh_name("share"),
                    inputs={
                        "value": src_op_name,
                        "setup": self.get_setup_op(dst_op.placement_name).name,
                    },
                    placement_name=dst_op.placement_name,
                    output_type=rep_dialect.ReplicatedRingTensorType(
                        dtype=src_op.output_type.dtype
                    ),
                )
            )
            self.incoming_edge_cache[cache_key] = share_op
        return self.incoming_edge_cache[cache_key]

    def process_outgoing_edge(self, src_op, input_key, dst_op_name):
        dst_op = self.computation.operation(dst_op_name)
        assert isinstance(dst_op, fixed_dialect.FixedpointOperation)

        cache_key = (src_op.name, dst_op.placement_name)
        if cache_key not in self.outgoing_edge_cache:
            reveal_op = self.computation.add_operation(
                rep_dialect.RevealOperation(
                    name=self.context.get_fresh_name("reveal"),
                    inputs={
                        "value": src_op.name,
                        "setup": self.get_setup_op(src_op.placement_name).name,
                    },
                    placement_name=src_op.placement_name,
                    recipient_name=dst_op.placement_name,
                    output_type=fixed_dialect.EncodedTensorType(
                        dtype=src_op.output_type.dtype,
                        precision=src_op.output_type.dtype.fractional_precision,
                    ),
                )
            )
            self.outgoing_edge_cache[cache_key] = reveal_op
        return self.outgoing_edge_cache[cache_key].name