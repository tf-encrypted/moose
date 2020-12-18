from moose.computation import replicated as replicated_ops
from moose.computation.replicated import EncodedTensorType
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.replicated import ReplicatedTensorType
from moose.computation.standard import TensorType


class ReplicatedShareRevealPass:
    def run(self, computation, context):
        # find edges to replicated placements from other placements
        share_edges = []
        for dst_op in computation.operations.values():
            dst_placement = computation.placement(dst_op.placement_name)
            if not isinstance(dst_placement, ReplicatedPlacement):
                continue
            for input_key, src_op_name in dst_op.inputs.items():
                src_op = computation.operation(src_op_name)
                src_placement = computation.placement(src_op.placement_name)
                if isinstance(src_placement, ReplicatedPlacement):
                    continue
                share_edges += [(src_op.name, dst_op.name, input_key)]

        # find edges from replicated placements to other placements
        reveal_edges = []
        for dst_op in computation.operations.values():
            dst_placement = computation.placement(dst_op.placement_name)
            if isinstance(dst_placement, ReplicatedPlacement):
                continue
            for input_key, src_op_name in dst_op.inputs.items():
                src_op = computation.operation(src_op_name)
                src_placement = computation.placement(src_op.placement_name)
                if not isinstance(src_placement, ReplicatedPlacement):
                    continue
                reveal_edges += [(src_op.name, dst_op.name, input_key)]

        # insert share operations where needed
        share_cache = dict()
        for (src_op_name, dst_op_name, input_key) in share_edges:
            src_op = computation.operation(src_op_name)
            dst_op = computation.operation(dst_op_name)

            # NOTE(Morten) assume that name of replicated placements is their identity
            # TODO(Morten) verify everywhere that diff placement name => diff setup
            cache_key = (src_op.name, dst_op.placement_name)

            if cache_key not in share_cache:
                datatype = {"float": "fixed64"}[src_op.output_type.datatype]
                encode_op = replicated_ops.EncodeOperation(
                    name=context.get_fresh_name("encode"),
                    inputs={"value": src_op.name},
                    placement_name=dst_op.placement_name,
                    scaling_factor=2 ** 16,
                    output_type=EncodedTensorType(datatype=datatype),
                )
                share_op = replicated_ops.ShareOperation(
                    name=context.get_fresh_name("share"),
                    inputs={"value": encode_op.name, "setup": dst_op.inputs["setup"]},
                    placement_name=dst_op.placement_name,
                    output_type=ReplicatedTensorType(datatype=datatype),
                )
                computation.add_operation(encode_op)
                computation.add_operation(share_op)
                share_cache[cache_key] = share_op

            share_op = share_cache[cache_key]
            dst_op.inputs[input_key] = share_op.name

        reveal_cache = dict()
        for (src_op_name, dst_op_name, input_key) in reveal_edges:
            src_op = computation.operation(src_op_name)
            dst_op = computation.operation(dst_op_name)

            cache_key = (src_op.name, dst_op.placement_name)
            if cache_key not in reveal_cache:
                reveal_op = replicated_ops.RevealOperation(
                    name=context.get_fresh_name("reveal"),
                    inputs={"value": src_op.name, "setup": src_op.inputs["setup"]},
                    placement_name=src_op.placement_name,
                    recipient_name=dst_op.placement_name,
                    output_type=EncodedTensorType(datatype=src_op.output_type.datatype),
                )
                datatype = {"fixed64": "float"}[src_op.output_type.datatype]
                decode_op = replicated_ops.DecodeOperation(
                    name=context.get_fresh_name("decode"),
                    inputs={"value": reveal_op.name},
                    placement_name=src_op.placement_name,
                    output_type=TensorType(datatype=datatype),
                    scaling_factor=2 ** 16,
                )
                computation.add_operation(reveal_op)
                computation.add_operation(decode_op)
                reveal_cache[cache_key] = decode_op

            reveal_op = reveal_cache[cache_key]
            dst_op.inputs[input_key] = reveal_op.name

        return computation, True
