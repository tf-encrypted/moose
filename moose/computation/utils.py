import marshal
from dataclasses import asdict

import moose.computation.host
import moose.computation.mpspdz
import moose.computation.standard
from moose.computation.base import Computation
from moose.computation.base import Graph


def serialize_computation(computation):
    return marshal.dumps(asdict(computation))


def deserialize_computation(bytes_stream):
    computation_dict = marshal.loads(bytes_stream)
    nodes_dict = computation_dict["graph"]["nodes"]
    nodes = {node: select_op(node)(**args) for node, args in nodes_dict.items()}
    return Computation(Graph(nodes))


def select_op(op_name):
    name = op_name.split("_")[:-1]
    name = "".join([n.title() for n in name]) + "Operation"
    for module in [moose.compiler.standard, moose.compiler.host, moose.compiler.mpspdz]:
        op = getattr(module, name, None)
        if op:
            return op
    raise ValueError(f"Failed to map operation '{op_name}'")
