from pymoose.computation.host import HostPlacement
from pymoose.computation.standard import DeserializeOperation
from pymoose.computation.standard import ReceiveOperation
from pymoose.computation.standard import SendOperation
from pymoose.computation.standard import SerializeOperation


class NetworkingPass:
    def __init__(self, reuse_when_possible=True):
        self.reuse_when_possible = reuse_when_possible
        self.serialize_cache = dict()
        self.deserialize_cache = dict()

    def run(self, computation, context):
        # we first find all edges to cut since we cannot mutate dict while traversing
        edges_to_cut = []
        for dst_op in computation.operations.values():
            dst_placement = computation.placement(dst_op.placement_name)
            if not isinstance(dst_placement, HostPlacement):
                continue
            for input_key, input_name in dst_op.inputs.items():
                src_op = computation.operation(input_name)
                if src_op.placement_name != dst_op.placement_name:
                    edges_to_cut += [(src_op, dst_op, input_key)]

        # cut each edge and replace with networking ops
        # we keep a cache of certain ops to avoid redundancy
        performed_changes = False
        for src_op, dst_op, input_key in edges_to_cut:
            patched_src_op, extra_ops = self.add_networking(
                context, src_op, dst_op.placement_name
            )
            computation.add_operations(extra_ops)
            dst_op.inputs[input_key] = patched_src_op.name
            performed_changes = True

        return computation, performed_changes

    def add_networking(self, context, source_operation, destination_placement_name):
        extra_ops = []

        if source_operation.placement_name == destination_placement_name:
            # nothing to do, we are already on the same placement
            return source_operation, extra_ops

        derialize_cache_key = (source_operation.name, destination_placement_name)
        if self.reuse_when_possible and derialize_cache_key in self.deserialize_cache:
            # nothing to do, we can reuse everything
            return self.deserialize_cache[derialize_cache_key], extra_ops

        # maybe we can reuse the serialized value
        serialize_cache_key = (source_operation.name, source_operation.placement_name)
        if self.reuse_when_possible and serialize_cache_key in self.serialize_cache:
            serialize_operation = self.serialize_cache[serialize_cache_key]
        else:
            serialize_operation = SerializeOperation(
                placement_name=source_operation.placement_name,
                name=context.get_fresh_name("serialize"),
                inputs={"value": source_operation.name},
                output_type=source_operation.output_type,
            )
            self.serialize_cache[serialize_cache_key] = serialize_operation
            extra_ops += [serialize_operation]

        rendezvous_key = context.get_fresh_name("rdv_key")
        send_operation = SendOperation(
            placement_name=source_operation.placement_name,
            name=context.get_fresh_name("send"),
            inputs={"value": serialize_operation.name},
            sender=source_operation.placement_name,
            receiver=destination_placement_name,
            rendezvous_key=rendezvous_key,
        )
        receive_operation = ReceiveOperation(
            placement_name=destination_placement_name,
            name=context.get_fresh_name("receive"),
            inputs={},
            sender=source_operation.placement_name,
            receiver=destination_placement_name,
            rendezvous_key=rendezvous_key,
            output_type=source_operation.output_type,
        )
        deserialize_operation = DeserializeOperation(
            placement_name=destination_placement_name,
            name=context.get_fresh_name("deserialize"),
            inputs={"value": receive_operation.name},
            output_type=source_operation.output_type,
        )
        self.deserialize_cache[derialize_cache_key] = deserialize_operation
        extra_ops += [send_operation, receive_operation, deserialize_operation]
        return deserialize_operation, extra_ops