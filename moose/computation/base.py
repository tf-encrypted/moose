from dataclasses import dataclass
from typing import Dict


@dataclass
class Placement:
    name: str

    def __enter__(self):
        global CURRENT_PLACEMENT
        CURRENT_PLACEMENT.append(self)

    def __exit__(self, type, value, traceback):
        global CURRENT_PLACEMENT
        CURRENT_PLACEMENT.pop(-1)

    def __hash__(self):
        return hash(self.name)

    def compile(self, context, fn, inputs, output_placements=None):
        raise NotImplementedError()


@dataclass
class Operation:
    placement_name: str
    name: str
    inputs: Dict[str, str]

    @classmethod
    def identifier(cls):
        return cls.__name__


@dataclass
class Graph:
    nodes: Dict[str, Operation]


@dataclass
class Computation:
    graph: Graph
    placements: Dict[str, Placement]

    def placements(self):
        return set(node.placement for node in self.graph.nodes.values())

    def operations(self):
        return self.graph.nodes.values()

    def operations_of_type(self, op_type):
        return [op for op in self.operations() if isinstance(op, op_type)]

    def operation(self, name):
        return self.graph.nodes.get(name)

    def add_operation(self, op):
        assert op.name not in self.graph.nodes, op.name
        self.graph.nodes[op.name] = op

    def remove_operation(self, op):
        del self.graph.nodes[op.name]

    def replace_operation(self, old_op, new_op):
        assert new_op.name == old_op.name
        self.remove_operation(old_op)
        self.add_operation(new_op)

    def add_operations(self, ops):
        for op in ops:
            self.add_operation(op)
