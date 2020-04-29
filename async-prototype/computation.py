from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Device:
    name: str


@dataclass
class NativeDevice(Device):
    pass


@dataclass
class Operation:
    device: Device
    name: str
    inputs: Dict[str, str]
    outputs: List[str]


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


in0_device = NativeDevice(name="inputter0")
in1_device = NativeDevice(name="inputter1")
agg_device = NativeDevice(name="aggregator")
out_device = NativeDevice(name="outputter")


ops = [
    LoadOperation(
        device=in0_device, name="load0", key="x0", inputs={}, outputs=["x0_on_in0"]
    ),
    LoadOperation(
        device=in1_device, name="load1", key="x1", inputs={}, outputs=["x1_on_in1"]
    ),
    SendOperation(
        device=in0_device,
        name="send0",
        inputs={"value": "x0_on_in0"},
        outputs=[],
        channel="in0_agg",
        rendezvous_key="x0",
    ),
    SendOperation(
        device=in1_device,
        name="send1",
        inputs={"value": "x1_on_in1"},
        outputs=[],
        channel="in1_agg",
        rendezvous_key="x1",
    ),
    ReceiveOperation(
        device=agg_device,
        name="recv0",
        channel="in0_agg",
        rendezvous_key="x0",
        inputs={},
        outputs=["x0_on_agg"],
    ),
    ReceiveOperation(
        device=agg_device,
        name="recv1",
        channel="in1_agg",
        rendezvous_key="x1",
        inputs={},
        outputs=["x1_on_agg"],
    ),
    AddOperation(
        device=agg_device,
        name="add",
        inputs={"lhs": "x0_on_agg", "rhs": "x1_on_agg"},
        outputs=["y_on_agg"],
    ),
    SendOperation(
        device=agg_device,
        name="send",
        inputs={"value": "y_on_agg"},
        outputs=[],
        channel="agg_out",
        rendezvous_key="y",
    ),
    ReceiveOperation(
        device=out_device,
        name="recv",
        channel="agg_out",
        rendezvous_key="y",
        inputs={},
        outputs=["y_on_out"],
    ),
    SaveOperation(
        device=out_device,
        name="save",
        key="y",
        inputs={"value": "y_on_out"},
        outputs=[],
    ),
]


comp = Computation(graph={op.name: op for op in ops})
