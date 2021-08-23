from abc import abstractmethod

from pymoose.deprecated.compiler.pruning import PruningPass


class SubgraphReplacementPass:
    """Generic pass for replacing subgraph in computation."""

    def __init__(self):
        self.computation = None
        self.context = None
        self.setup_cache = None
        self.op_names_to_lower = None

    @abstractmethod
    def collect_subgraph(self):
        pass

    @abstractmethod
    def process_incoming_edge(src_op_name, input_key, dst_op_name):
        pass

    @abstractmethod
    def process_output_edge(src_op, input_key, dst_op_name):
        pass

    def run(self, computation, context):
        self.node_cache = dict()
        self.computation = computation
        self.context = context

        # collect ops to process
        self.op_names_to_lower = self.collect_subgraph()

        # collect ops that we need to rewire afterwards
        outgoing_edges = set()
        for dst_op in self.computation.operations.values():
            if dst_op.name in self.op_names_to_lower:
                continue
            for input_key, src_op_name in dst_op.inputs.items():
                if src_op_name in self.op_names_to_lower:
                    outgoing_edges.add((src_op_name, input_key, dst_op.name))

        # process collected ops by adding new ops to the computations
        processed_cache = dict()
        for op_name in sorted(self.op_names_to_lower):
            processed_cache[op_name] = self.process_node(op_name)

        # rewire ops for outgoing edges
        for src_op_name, input_key, dst_op_name in sorted(outgoing_edges):
            dst_op = self.computation.operation(dst_op_name)
            processed_src_op = processed_cache[src_op_name]
            dst_op.inputs[input_key] = self.process_outgoing_edge(
                src_op=processed_src_op, input_key=input_key, dst_op_name=dst_op.name,
            )

        # prune old ops
        pruning_pass = PruningPass()
        computation, pruning_performed_changes = pruning_pass.run(computation, context)

        performed_changes = len(self.op_names_to_lower) > 0 or pruning_performed_changes
        return computation, performed_changes

    def process_node(self, op_name):
        cache_key = op_name
        if cache_key not in self.node_cache:
            op = self.computation.operation(op_name)
            processed_inputs = {
                input_key: self.process_edge(
                    src_op_name=input_op_name, input_key=input_key, dst_op_name=op.name,
                )
                for input_key, input_op_name in op.inputs.items()
            }
            processing_fn = getattr(self, f"process_{type(op).__name__}", None)
            if processing_fn is None:
                raise NotImplementedError(f"{type(op)}")
            self.node_cache[cache_key] = processing_fn(op, processed_inputs)
        return self.node_cache[cache_key]

    def process_edge(self, src_op_name, input_key, dst_op_name):
        if src_op_name not in self.op_names_to_lower:
            return self.process_incoming_edge(src_op_name, input_key, dst_op_name)
        else:
            return self.process_node(src_op_name)
