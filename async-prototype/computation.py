from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional


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
class AddOperation(Operation):
    pass


@dataclass
class SendOperation(Operation):
    channel: str
    rendezvous_key: str


@dataclass
class ReceiveOperation(Operation):
    channel: str
    rendezvous_key: str


@dataclass
class Computation:
    graph: Dict[str, Operation]

    def devices(self):
        return set(node.device for node in self.graph)

    def nodes(self):
        return self.graph.values()

    def node(self, name):
        return self.graph.get(name)
