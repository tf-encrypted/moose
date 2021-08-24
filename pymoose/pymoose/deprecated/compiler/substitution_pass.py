import abc

from pymoose.deprecated.compiler import pruning


class SubstitutionPass(metaclass=abc.ABCMeta):
    """A generic compiler pass for substituting individual ops with subgraphs.

    Ops are qualified by the pass and then lowered according to op-specific lowering
    instructions. Note that this pass is only concerned with lowering single ops into
    subgraphs. For lowering subgraphs into subgraphs, see SubgraphReplacementPass.
    """

    def __init__(self):
        self.computation = None
        self.context = None

    @abc.abstractmethod
    def qualify_substitution(self, op):
        """Determines if op should be lowered by this pass.

        If so, the pass must have an op-specific lowering instruction.
        """
        pass

    @abc.abstractmethod
    def lower(self, op_name):
        """Finds and performs the lowering instruction for a qualified op."""
        pass

    def run(self, computation, context):
        self.computation = computation
        self.context = context

        # collect ops to lower via substitution
        op_names_to_lower = set()
        for op in computation.operations.values():
            qualified = self.qualify_substitution(op)
            if qualified:
                op_names_to_lower.add(op.name)

        # lower ops
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
        pruning_pass = pruning.PruningPass()
        computation, pruning_performed_changes = pruning_pass.run(computation, context)

        # if we changed the graph at all, let the compiler know
        performed_changes = len(op_names_to_lower) > 0 or pruning_performed_changes
        return computation, performed_changes

    def _rewire_output_ops(self, old_src_op, new_src_op):
        """Rewire the output edges of old_src_op to new_src_op."""
        dst_ops = self.computation.find_destinations(old_src_op)
        for dst_op in dst_ops:
            updated_wirings = {
                k: new_src_op.name
                for k, v in dst_op.inputs.items()
                if v == old_src_op.name
            }
            dst_op.inputs.update(updated_wirings)
