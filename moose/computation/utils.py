import marshal
from dataclasses import asdict

import moose.computation.host
import moose.computation.mpspdz
import moose.computation.standard
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.host import HostPlacement
from moose.computation.mpspdz import MpspdzPlacement
from moose.computation.replicated import ReplicatedPlacement


def serialize_computation(computation):
    return marshal.dumps(asdict(computation))


def deserialize_computation(bytes_stream):
    computation_dict = marshal.loads(bytes_stream)
    operations = {
        op_name: select_op(op_name)(**args)
        for op_name, args in computation_dict["operations"].items()
    }
    placements = {
        plc_name: select_plc(args)
        for plc_name, args in computation_dict["placements"].items()
    }
    return Computation(operations=operations, placements=placements)


def select_op(op_name):
    name = op_name.split("_")[:-1]
    name = "".join([n.title() for n in name]) + "Operation"
    for module in [
        moose.computation.standard,
        moose.computation.host,
        moose.computation.mpspdz,
    ]:
        op = getattr(module, name, None)
        if op:
            assert issubclass(op, Operation)
            return op
    raise ValueError(f"Failed to map operation '{op_name}'")


def select_plc(args):
    plc_ty = {
        "host": HostPlacement,
        "mpspdz": MpspdzPlacement,
        "replicated": ReplicatedPlacement,
    }[args["ty"]]
    return plc_ty(**args)
