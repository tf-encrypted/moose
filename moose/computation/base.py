from dataclasses import dataclass
from typing import Dict


@dataclass
class Placement:
    name: str

    def __hash__(self):
        return hash(self.name)


@dataclass
class Operation:
    placement_name: str
    name: str
    inputs: Dict[str, str]

    @classmethod
    def identifier(cls):
        return cls.__name__


@dataclass
class Computation:
    operations: Dict[str, Operation]
    placements: Dict[str, Placement]

    def find_destinations(self, op):
        destination_ops = []
        for candidate_op in self.operations.values():
            if op.name in candidate_op.inputs.values():
                destination_ops += [candidate_op]
        return destination_ops

    def find_sources(self, op):
        source_ops = []
        for input_op_name in op.inputs.values():
            op = self.operation(input_op_name)
            source_ops += [op]
        return source_ops

    def placement(self, name):
        return self.placements.get(name)

    def add_placement(self, placement):
        assert isinstance(placement, Placement)
        assert placement.name not in self.placements
        self.placements[placement.name] = placement
        return placement

    def maybe_add_placement(self, placement):
        if placement.name in self.placements:
            assert placement == self.placements[placement.name]
            return placement
        return self.add_placement(placement)

    def find_operations_of_type(self, op_type):
        return [op for op in self.operations.values() if isinstance(op, op_type)]

    def operation(self, name):
        return self.operations.get(name)

    def add_operation(self, op):
        assert isinstance(op, Operation)
        assert op.name not in self.operations, op.name
        assert op.placement_name in self.placements, op.placement_name
        self.operations[op.name] = op
        return op

    def add_operations(self, ops):
        for op in ops:
            self.add_operation(op)

    def remove_operation(self, op):
        del self.operations[op.name]

    def replace_operation(self, old_op, new_op):
        assert new_op.name == old_op.name
        self.remove_operation(old_op)
        self.add_operation(new_op)
