from moose.compiler.replicated.subgraph_replace_pass import SubgraphReplacementPass
from moose.computation import fixedpoint as fixed_dialect
from moose.computation import replicated as rep_dialect


class ReplicatedOpsPass(SubgraphReplacementPass):
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
            if not isinstance(op_placement, rep_dialect.ReplicatedPlacement):
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
                output_type=rep_dialect.ReplicatedTensorType(
                    datatype=op.output_type.datatype
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
                output_type=rep_dialect.ReplicatedTensorType(
                    datatype=op.output_type.datatype
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
                output_type=rep_dialect.ReplicatedTensorType(
                    datatype=op.output_type.datatype
                ),
            )
        )

    def process_EncodeOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.EncodeOperation)
        encode_op = self.computation.add_operation(
            fixed_dialect.EncodeOperation(
                name=self.context.get_fresh_name("encode"),
                inputs={
                    input_key: input_op.name
                    for input_key, input_op in processed_inputs.items()
                },
                placement_name=op.placement_name,
                precision=op.precision,
                output_type=op.output_type,
            )
        )
        share_op = self.computation.add_operation(
            rep_dialect.ShareOperation(
                name=self.context.get_fresh_name("share"),
                inputs={
                    "value": encode_op.name,
                    "setup": self.get_setup_op(encode_op.placement_name).name,
                },
                placement_name=encode_op.placement_name,
                output_type=rep_dialect.ReplicatedTensorType(
                    datatype=encode_op.output_type.datatype
                ),
            )
        )
        return share_op

    def process_DecodeOperation(self, op, processed_inputs):
        assert isinstance(op, fixed_dialect.DecodeOperation)
        decode_op = self.computation.add_operation(
            fixed_dialect.DecodeOperation(
                name=self.context.get_fresh_name("decode"),
                inputs={
                    input_key: input_op.name
                    for input_key, input_op in processed_inputs.items()
                },
                placement_name=op.placement_name,
                precision=op.precision,
                output_type=op.output_type,
            )
        )
        return decode_op

    def process_incoming_edge(self, src_op_name, input_key, dst_op_name):
        return self.computation.operation(src_op_name)

    def process_outgoing_edge(self, src_op, input_key, dst_op_name):
        assert isinstance(src_op, fixed_dialect.DecodeOperation)
        dst_op = self.computation.operation(dst_op_name)
        src_input_op = self.computation.operation(src_op.inputs["value"])

        cache_key = (src_op.name, dst_op.placement_name)
        if cache_key not in self.outgoing_edge_cache:
            reveal_op = self.computation.add_operation(
                rep_dialect.RevealOperation(
                    name=self.context.get_fresh_name("reveal"),
                    inputs={
                        "value": src_input_op.name,
                        "setup": self.get_setup_op(src_op.placement_name).name,
                    },
                    placement_name=src_op.placement_name,
                    recipient_name=dst_op.placement_name,
                    output_type=fixed_dialect.EncodedTensorType(
                        datatype=src_input_op.output_type.datatype,
                        precision=src_op.precision,
                    ),
                )
            )
            decode_op = self.computation.add_operation(
                fixed_dialect.DecodeOperation(
                    name=self.context.get_fresh_name("decode"),
                    inputs={"value": reveal_op.name},
                    placement_name=src_op.placement_name,
                    precision=src_op.precision,
                    output_type=src_op.output_type,
                )
            )
            self.outgoing_edge_cache[cache_key] = decode_op
        return self.outgoing_edge_cache[cache_key].name
