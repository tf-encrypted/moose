from moose.computations.standard import OutputOperation


class PruningPass:
    def run(self, computation, context):
        performed_changes = False
        ops_to_keep = set()

        # iteratively expand set of operations to keep
        frontier = set(
            op_name
            for op_name, op in computation.operations.items()
            if isinstance(op, OutputOperation)
        )
        while True:
            ops_to_keep.update(frontier)
            new_frontier = set()
            for op_name in frontier:
                op = computation.operation(op_name)
                new_frontier.update(
                    input_op_name for input_op_name in op.inputs.values()
                )
            if not new_frontier:
                # nothing was added so done expanding
                break
            frontier = new_frontier

        # remove all operations we didn't visit
        ops_to_remove = set(computation.operations.keys()) - ops_to_keep
        if ops_to_remove:
            computation.remove_operations(ops_to_remove)
            performed_changes = True

        return computation, performed_changes
