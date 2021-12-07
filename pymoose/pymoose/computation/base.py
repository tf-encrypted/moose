from dataclasses import dataclass
from dataclasses import field
from typing import Dict


@dataclass
class Placement:
    name: str

    def __hash__(self):
        return hash(self.name)


@dataclass
class ValueType:
    pass


@dataclass
class Value:
    pass


@dataclass
class OpSignature:
    input_types: Dict[str, ValueType]
    return_type: ValueType


@dataclass(init=False)
class Operation:
    name: str
    inputs: Dict[str, str]
    placement_name: str
    signature: OpSignature

    @classmethod
    def identifier(cls):
        return cls.__name__

    @property
    def return_type(self):
        return self.signature.return_type


@dataclass
class Computation:
    operations: Dict[str, Operation] = field(default_factory=dict)
    placements: Dict[str, Placement] = field(default_factory=dict)

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

    def add(self, component):
        if isinstance(component, Operation):
            return self.add_operation(component)
        if isinstance(component, Placement):
            return self.add_placement(component)
        raise NotImplementedError(f"{component}")

    def maybe_add(self, component):
        if isinstance(component, Operation):
            return self.maybe_add_operation(component)
        if isinstance(component, Placement):
            return self.maybe_add_placement(component)
        raise NotImplementedError(f"{component}")

    def placement(self, name):
        return self.placements[name]

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
        return self.operations[name]

    def add_operation(self, op):
        assert isinstance(op, Operation)
        assert op.name not in self.operations, op
        assert op.placement_name in self.placements, op.placement_name
        self.operations[op.name] = op
        return op

    def maybe_add_operation(self, op):
        assert isinstance(op, Operation)
        if op.name in self.operations:
            assert op == self.operations[op.name]
            return op
        return self.add_operation(op)

    def add_operations(self, ops):
        for op in ops:
            self.add_operation(op)

    def remove_operation(self, name):
        del self.operations[name]

    def remove_operations(self, names):
        for name in names:
            self.remove_operation(name)

    def rewire(self, old_op, new_op):
        assert old_op.name in self.operations, old_op
        assert new_op.name in self.operations, new_op
        for op in self.operations.values():
            for arg in op.inputs.keys():
                if op.inputs[arg] == old_op.name:
                    op.inputs[arg] = new_op.name
