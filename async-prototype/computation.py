import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Union

OPS_REGISTER = {}


@dataclass
class Operation:
    device_name: str
    name: str
    inputs: Dict[str, str]
    output: Optional[str]


@dataclass
class LoadOperation(Operation):
    key: str


@dataclass
class SaveOperation(Operation):
    key: str


@dataclass
class ConstantOperation(Operation):
    value: Union[int, float]


@dataclass
class AddOperation(Operation):
    pass


@dataclass
class SubOperation(Operation):
    pass


@dataclass
class MulOperation(Operation):
    pass


@dataclass
class SendOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str


@dataclass
class ReceiveOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str


@dataclass
class Graph:
    nodes: Dict[str, Operation]


@dataclass
class Computation:
    graph: Graph

    def devices(self):
        return set(node.device for node in self.graph.nodes.values())

    def nodes(self):
        return self.graph.nodes.values()

    def node(self, name):
        return self.graph.nodes.get(name)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(self, computation_dict):
        nodes_dict = computation_dict["graph"]["nodes"]
        nodes = {}
        for node, args in nodes_dict.items():
            nodes[node] = select_op(node)(**args)
        computation = Computation(Graph(nodes))
        return computation

    def serialize(self):
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def deserialize(self, bytes_stream):
        return self.from_dict(json.loads(bytes_stream.decode("utf-8")))


def select_op(op_name):
    name = op_name.split("_")[0]
    if "operation" in name:
        name = re.sub("operation", "", name)
    name = name[0].upper() + name[1:] + "Operation"
    op = OPS_REGISTER[name]
    return op


def register_op(name, op):
    OPS_REGISTER[name] = op


register_op("AddOperation", AddOperation)
register_op("LoadOperation", LoadOperation)
register_op("ConstantOperation", ConstantOperation)
register_op("MulOperation", MulOperation)
register_op("SaveOperation", SaveOperation)
register_op("SendOperation", SendOperation)
register_op("SubOperation", SubOperation)
register_op("ReceiveOperation", ReceiveOperation)
