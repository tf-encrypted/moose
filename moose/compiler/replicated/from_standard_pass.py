from moose.computation import replicated as replicated_ops
from moose.computation import standard as standard_ops
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.replicated import ReplicatedTensorType
from moose.computation.standard import StandardOperation


class ReplicatedFromStandardOpsPass:
    # This pass lowers all standard ops on replicated placements to their
    # corresponding replicated ops, adding setup where needed.

    def __init__(self):
        self.computation = None
        self.context = None
        self.setup_cache = None

    def run(self, computation, context):
        self.computation = computation
        self.context = context
        self.setup_cache = dict()

        ops_to_lower = []
        for op in self.computation.operations.values():
            if not isinstance(op, StandardOperation):
                continue
            op_placement = self.computation.placement(op.placement_name)
            if not isinstance(op_placement, ReplicatedPlacement):
                continue
            ops_to_lower += [op.name]

        for op_name in ops_to_lower:
            self.process(op_name)
        return self.computation, len(ops_to_lower) > 0

    def process(self, op_name):
        # process based on op type
        op = self.computation.operation(op_name)
        process_fn = getattr(self, f"process_{type(op).__name__}", None)
        if process_fn is None:
            raise NotImplementedError(f"{type(op)}")
        process_fn(op)

    def get_setup_op(self, placement_name):
        cache_key = placement_name
        if cache_key not in self.setup_cache:
            setup_op = replicated_ops.SetupOperation(
                name=self.context.get_fresh_name("replicated_setup"),
                placement_name=placement_name,
                inputs={},
            )
            self.computation.add_operation(setup_op)
            self.setup_cache[cache_key] = setup_op
        return self.setup_cache[cache_key]

    def process_AddOperation(self, op):
        assert isinstance(op, standard_ops.AddOperation)
        new_inputs = op.inputs.copy()
        assert "setup" not in new_inputs
        new_inputs["setup"] = self.get_setup_op(op.placement_name).name
        new_datatype = {"float": "fixed64"}[op.output_type.datatype]
        new_op = replicated_ops.AddOperation(
            name=self.context.get_fresh_name("replicated_add"),
            placement_name=op.placement_name,
            inputs=new_inputs,
            output_type=ReplicatedTensorType(datatype=new_datatype),
        )
        self.computation.add_operation(new_op)
        self.computation.rewire(op, new_op)
        self.computation.remove_operation(op.name)

    def process_MulOperation(self, op):
        assert isinstance(op, standard_ops.MulOperation)
        new_inputs = op.inputs.copy()
        assert "setup" not in new_inputs
        new_inputs["setup"] = self.get_setup_op(op.placement_name).name
        new_datatype = {"float": "fixed64"}[op.output_type.datatype]
        new_op = replicated_ops.MulOperation(
            name=self.context.get_fresh_name("replicated_mul"),
            placement_name=op.placement_name,
            inputs=new_inputs,
            output_type=ReplicatedTensorType(datatype=new_datatype),
        )
        self.computation.add_operation(new_op)
        self.computation.rewire(op, new_op)
        self.computation.remove_operation(op.name)

    def process_DotOperation(self, op):
        assert isinstance(op, standard_ops.DotOperation)
        new_inputs = op.inputs.copy()
        assert "setup" not in new_inputs
        new_inputs["setup"] = self.get_setup_op(op.placement_name).name
        new_datatype = {"float": "fixed64"}[op.output_type.datatype]
        new_op = replicated_ops.DotOperation(
            name=self.context.get_fresh_name("replicated_dot"),
            placement_name=op.placement_name,
            inputs=new_inputs,
            output_type=ReplicatedTensorType(datatype=new_datatype),
        )
        self.computation.add_operation(new_op)
        self.computation.rewire(op, new_op)
        self.computation.remove_operation(op.name)
