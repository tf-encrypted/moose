import inspect
import marshal
from dataclasses import asdict

import moose.computation.host
import moose.computation.mpspdz
import moose.computation.replicated
import moose.computation.standard
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.logger import get_logger


def serialize_computation(computation):
    return marshal.dumps(asdict(computation))


def deserialize_computation(bytes_stream):
    computation_dict = marshal.loads(bytes_stream)
    get_logger().debug(computation_dict)
    operations = {
        op_name: select_op(args)
        for op_name, args in computation_dict["operations"].items()
    }
    placements = {
        plc_name: select_plc(args)
        for plc_name, args in computation_dict["placements"].items()
    }
    return Computation(operations=operations, placements=placements)


_known_ops_cache = None


def known_ops():
    global _known_ops_cache
    if _known_ops_cache is None:
        _known_ops_cache = dict()
        for module in [
            moose.computation.standard,
            moose.computation.host,
            moose.computation.mpspdz,
        ]:
            for clazz_name, clazz in inspect.getmembers(module, inspect.isclass):
                if clazz is Operation:
                    continue
                if not issubclass(clazz, Operation):
                    continue
                ty = getattr(clazz, "ty", None)
                if not ty:
                    get_logger().warning(
                        f"Ignoring operation without 'ty' field; op:{clazz_name}"
                    )
                    continue
                if ty in _known_ops_cache:
                    get_logger().warning(
                        f"Ignoring duplicate operation;"
                        f" op1:{clazz_name},"
                        f" op2:{_known_ops_cache[ty]}"
                    )
                    continue
                _known_ops_cache[ty] = clazz
    return _known_ops_cache


def select_op(args):
    assert "ty" in args, args
    ops = known_ops()
    op_ty = ops.get(args["ty"], None)
    if not op_ty:
        raise ValueError(f"Failed to map operation; ty:'{args['ty']}'")
    return op_ty(**args)


_known_plcs_cache = None


def known_plcs():
    global _known_plcs_cache
    if _known_plcs_cache is None:
        _known_plcs_cache = dict()
        for module in [
            moose.computation.host,
            moose.computation.mpspdz,
            moose.computation.replicated,
        ]:
            for clazz_name, clazz in inspect.getmembers(module, inspect.isclass):
                if clazz is Placement:
                    continue
                if not issubclass(clazz, Placement):
                    continue
                ty = getattr(clazz, "ty", None)
                if not ty:
                    get_logger().warning(
                        f"Ignoring placement without 'ty' field; op:{clazz_name}"
                    )
                    continue
                if ty in _known_plcs_cache:
                    get_logger().warning(
                        f"Ignoring duplicate placement;"
                        f" op1:{clazz_name},"
                        f" op2:{_known_plcs_cache[ty]}"
                    )
                    continue
                _known_plcs_cache[ty] = clazz
    return _known_plcs_cache


def select_plc(args):
    assert "ty" in args, args
    plcs = known_plcs()
    plc_ty = plcs.get(args["ty"], None)
    if not plc_ty:
        raise ValueError(f"Failed to map placement; ty:'{args['ty']}'")
    return plc_ty(**args)
