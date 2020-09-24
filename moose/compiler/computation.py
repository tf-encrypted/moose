import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

OPS_REGISTER = {}


@dataclass
class Operation:
    device_name: str
    name: str
    inputs: Dict[str, str]
    output: Optional[str]

    @classmethod
    def identifier(cls):
        return cls.__name__


@dataclass
class InputOperation(Operation):
    # Represents an input to the computation that is already
    # available on the assigned placement. This operation takes no
    # inputs on its own.
    pass


@dataclass
class EnterOperation(Operation):
    # This is the mechanism by which a subcomputation is invoked.
    # There must be an enter operation for each value passing into the
    # subcomputation, potentially on overlapping placements, although
    # several inputs (on the same placement) may also come in via the
    # same enter (for extra control over strict vs lazy inputs).
    #
    # There must also be an enter operation for every placement that
    # takes part in the subcomputation or its sub-sub-computations:
    # entering an computation is what causes the placement's executor
    # to schedule the evaluation of kernels, so without entering a
    # computation an executor will never produce any expected outputs
    # and the computation will hang. The unit value is used to enter
    # a placement without explicit input.
    #
    # The session id of the subcomputation is derived from the parents
    # session id and the operation's activate key; this means that
    # activation keys must match for enter operations that belong to
    # the same computation "call". Activation keys are also used to
    # keep two "calls" to the same subcomputation distinct.
    #
    # Enter operations behave like Send operations and have no outputs;
    # all outputs passed back from the subcomputation is delivered via
    # ExitOperations.
    #
    # The computation id may refer to either a computation symbol or
    # a computation closure, but must be previously defined on the
    # executor.
    #
    # Non-unit enter operations are mapped to input operations on the
    # subcomputation.
    activation_key: str
    computation_id: str
    inputs_mapping: Dict


@dataclass
class ExitOperation(Operation):
    activation_key: str


@dataclass
class SendOperation(Operation):
    # Makes a local value from the session on the sender available
    # to the receiver.

    sender: str
    receiver: str
    rendezvous_key: str


@dataclass
class ReceiveOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str


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
class DivOperation(Operation):
    pass


@dataclass
class RunProgramOperation(Operation):
    path: str
    args: List[str]


@dataclass
class CallPythonFunctionOperation(Operation):
    pickled_fn: bytes


@dataclass
class Graph:
    nodes: Dict[str, Operation]


@dataclass
class Computation:
    graph: Graph

    def devices(self):
        return set(node.device_name for node in self.graph.nodes.values())

    def nodes(self):
        return self.graph.nodes.values()

    def node(self, name):
        return self.graph.nodes.get(name)

    def serialize(self):
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def deserialize(cls, bytes_stream):
        computation_dict = json.loads(bytes_stream.decode("utf-8"))
        nodes_dict = computation_dict["graph"]["nodes"]
        nodes = {node: select_op(node)(**args) for node, args in nodes_dict.items()}
        return Computation(Graph(nodes))


def register_op(op):
    OPS_REGISTER[op.identifier()] = op


def select_op(op_name):
    name = op_name.split("_")[0]
    if "operation" in name:
        name = re.sub("operation", "", name)
    name = name[0].upper() + name[1:] + "Operation"
    op = OPS_REGISTER[name]
    return op


# NOTE: this is only needed for gRPC so far
register_op(AddOperation)
register_op(CallPythonFunctionOperation)
register_op(RunProgramOperation)
register_op(LoadOperation)
register_op(ConstantOperation)
register_op(DivOperation)
register_op(MulOperation)
register_op(SaveOperation)
register_op(SendOperation)
register_op(SubOperation)
register_op(ReceiveOperation)
