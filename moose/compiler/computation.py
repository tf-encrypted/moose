import marshal
import re
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import moose.compiler.computation


@dataclass
class Operation:
    placement_name: str
    name: str
    inputs: Dict[str, str]
    output: Optional[str]

    @classmethod
    def identifier(cls):
        return cls.__name__


@dataclass
class AddOperation(Operation):
    pass


@dataclass
class CallPythonFunctionOperation(Operation):
    pickled_fn: bytes = field(repr=False)
    output_type: Optional


@dataclass
class ConstantOperation(Operation):
    value: Union[int, float]


@dataclass
class DeserializeOperation(Operation):
    value_type: str


@dataclass
class DivOperation(Operation):
    pass


@dataclass
class LoadOperation(Operation):
    key: str


@dataclass
class MulOperation(Operation):
    pass


@dataclass
class ReceiveOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str


@dataclass
class RunProgramOperation(Operation):
    path: str
    args: List[str]


@dataclass
class SaveOperation(Operation):
    key: str


@dataclass
class SubOperation(Operation):
    pass


@dataclass
class SendOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str


@dataclass
class SerializeOperation(Operation):
    value_type: str


@dataclass
class MpspdzSaveInputOperation(Operation):
    player_index: int
    invocation_key: str


@dataclass
class MpspdzCallOperation(Operation):
    num_players: int
    player_index: int
    mlir: str = field(repr=False)
    invocation_key: str
    coordinator: str
    protocol: str


@dataclass
class MpspdzLoadOutputOperation(Operation):
    player_index: int
    invocation_key: str


@dataclass
class Graph:
    nodes: Dict[str, Operation]


@dataclass
class Computation:
    graph: Graph

    def placements(self):
        return set(node.placement for node in self.graph.nodes.values())

    def nodes(self):
        return self.graph.nodes.values()

    def node(self, name):
        return self.graph.nodes.get(name)

    def serialize(self):
        return marshal.dumps(asdict(self))

    @classmethod
    def deserialize(cls, bytes_stream):
        computation_dict = marshal.loads(bytes_stream)
        nodes_dict = computation_dict["graph"]["nodes"]
        nodes = {node: select_op(node)(**args) for node, args in nodes_dict.items()}
        return Computation(Graph(nodes))


def select_op(op_name):
    # To handle addoperation_op0, muloperation_op0 etc.
    if "operation" in op_name:
        op_name = re.sub("operation", "", op_name)
    name = op_name.split("_")[:-1]
    name = "".join([n.title() for n in name]) + "Operation"
    op = getattr(moose.compiler.computation, name, None)
    if op is None:
        raise ValueError(f"Unknown Moose runtime operation '{name}'")
    return op
