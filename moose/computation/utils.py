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
from moose.computation.base import ValueType
from moose.logger import get_logger


def serialize_computation(computation):
    return marshal.dumps(asdict(computation))


def deserialize_computation(bytes_stream):
    computation_dict = marshal.loads(bytes_stream)
    get_logger().debug(computation_dict)
    types = {
        ty_name: select_type(args)
        for ty_name, args in computation_dict["types"].items()
    }
    operations = {
        op_name: select_op(args)
        for op_name, args in computation_dict["operations"].items()
    }
    placements = {
        plc_name: select_plc(args)
        for plc_name, args in computation_dict["placements"].items()
    }
    return Computation(types=types, operations=operations, placements=placements)


_known_types_cache = None


def known_types():
    global _known_types_cache
    if _known_types_cache is None:
        _known_types_cache = dict()
        for module in [
            moose.computation.base,
            moose.computation.standard,
            moose.computation.host,
            moose.computation.mpspdz,
        ]:
            for class_name, class_ in inspect.getmembers(module, inspect.isclass):
                if class_ is ValueType:
                    continue
                if not issubclass(class_, ValueType):
                    continue
                kind = getattr(class_, "kind", None)
                if not kind:
                    get_logger().warning(
                        f"Ignoring type without 'kind' field; op:{class_name}"
                    )
                    continue
                if kind in _known_types_cache:
                    get_logger().warning(
                        f"Ignoring duplicate type;"
                        f" op1:{class_name},"
                        f" op2:{_known_types_cache[kind]}"
                    )
                    continue
                _known_types_cache[kind] = class_
    return _known_types_cache


def select_type(args):
    assert "kind" in args, args
    types = known_types()
    ty_kind = types.get(args["kind"], None)
    if not ty_kind:
        raise ValueError(f"Failed to map type; kind:'{args['kind']}'")
    return ty_kind(**args)


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
            for class_name, class_ in inspect.getmembers(module, inspect.isclass):
                if class_ is Operation:
                    continue
                if not issubclass(class_, Operation):
                    continue
                type_ = getattr(class_, "type_", None)
                if not type_:
                    get_logger().warning(
                        f"Ignoring operation without 'type_' field; op:{class_name}"
                    )
                    continue
                if type_ in _known_ops_cache:
                    get_logger().warning(
                        f"Ignoring duplicate operation;"
                        f" op1:{class_name},"
                        f" op2:{_known_ops_cache[type_]}"
                    )
                    continue
                _known_ops_cache[type_] = class_
    return _known_ops_cache


def select_op(args):
    assert "type_" in args, args
    ops = known_ops()
    op_type = ops.get(args["type_"], None)
    if not op_type:
        raise ValueError(f"Failed to map operation; type:'{args['type_']}'")
    return op_type(**args)


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
            for class_name, class_ in inspect.getmembers(module, inspect.isclass):
                if class_ is Placement:
                    continue
                if not issubclass(class_, Placement):
                    continue
                type_ = getattr(class_, "type_", None)
                if not type_:
                    get_logger().warning(
                        f"Ignoring placement without 'type_' field; op:{class_name}"
                    )
                    continue
                if type_ in _known_plcs_cache:
                    get_logger().warning(
                        f"Ignoring duplicate placement;"
                        f" op1:{class_name},"
                        f" op2:{_known_plcs_cache[type_]}"
                    )
                    continue
                _known_plcs_cache[type_] = class_
    return _known_plcs_cache


def select_plc(args):
    assert "type_" in args, args
    plcs = known_plcs()
    plc_type = plcs.get(args["type_"], None)
    if not plc_type:
        raise ValueError(f"Failed to map placement; type:'{args['type_']}'")
    return plc_type(**args)
