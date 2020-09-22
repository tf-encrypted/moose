from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import dill

from moose.compiler.computation import AddOperation
from moose.compiler.computation import CallPythonFunctionOperation
from moose.compiler.computation import Computation
from moose.compiler.computation import ConstantOperation
from moose.compiler.computation import DeserializeOperation
from moose.compiler.computation import DivOperation
from moose.compiler.computation import Graph
from moose.compiler.computation import LoadOperation
from moose.compiler.computation import MulOperation
from moose.compiler.computation import Operation
from moose.compiler.computation import ReceiveOperation
from moose.compiler.computation import RunProgramOperation
from moose.compiler.computation import SaveOperation
from moose.compiler.computation import SendOperation
from moose.compiler.computation import SerializeOperation
from moose.compiler.computation import SubOperation
from moose.logger import get_logger
from moose.runtime import get_runtime

CURRENT_PLACEMENT: List = []


@dataclass
class Placement:
    name: str

    def __enter__(self):
        global CURRENT_PLACEMENT
        CURRENT_PLACEMENT.append(self)

    def __exit__(self, type, value, traceback):
        global CURRENT_PLACEMENT
        CURRENT_PLACEMENT.pop(-1)

    def __hash__(self):
        return hash(self.name)

    def compile(self, context, fn, inputs, output_placements=None):
        raise NotImplementedError()


@dataclass
class HostPlacement(Placement):
    def __hash__(self):
        return hash(self.name)

    def compile(self, context, fn, inputs, output_placements=None, output_type=None):
        return CallPythonFunctionOperation(
            device_name=self.name,
            name=context.get_fresh_name("call_python_function_op"),
            pickled_fn=dill.dumps(fn),
            inputs=inputs,
            output=context.get_fresh_name("call_python_function"),
            output_type=output_type,
        )


@dataclass
class MpspdzPlacement(Placement):
    players: List[HostPlacement]

    def __hash__(self):
        return hash(self.name)

    def compile(self, context, fn, inputs, output_placements=None):
        # TODO(Morten)
        # This will likely emit call operations for two or more placements,
        # together with either the .mpc file or bytecode needed for the
        # MP-SPDZ runtime (bytecode is probably best)
        get_logger().debug(f"Inputs: {inputs}")
        get_logger().debug(f"Output placements: {output_placements}")
        raise NotImplementedError()


def get_current_placement():
    global CURRENT_PLACEMENT
    return CURRENT_PLACEMENT[-1]


@dataclass
class Expression:
    placement: Placement
    inputs: List

    def __hash__(self):
        return id(self)


@dataclass
class BinaryOpExpression(Expression):
    op_type: Operation

    def __hash__(self):
        return id(self)


@dataclass
class ApplyFunctionExpression(Expression):
    fn: Callable
    output_placements: Optional[List[Placement]]
    output_type: Optional

    def __hash__(self):
        return id(self)


@dataclass
class ConstantExpression(Expression):
    value: Union[int, float]

    def __hash__(self):
        return id(self)


@dataclass
class LoadExpression(Expression):
    key: str

    def __hash__(self):
        return id(self)


@dataclass
class RunProgramExpression(Expression):
    path: str
    args: List[str]

    def __hash__(self):
        return id(self)


@dataclass
class SaveExpression(Expression):
    key: str

    def __hash__(self):
        return id(self)


def add(lhs, rhs):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    return BinaryOpExpression(
        op_type=AddOperation, placement=get_current_placement(), inputs=[lhs, rhs]
    )


def constant(value):
    return ConstantExpression(placement=get_current_placement(), inputs=[], value=value)


def div(lhs, rhs):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    return BinaryOpExpression(
        op_type=DivOperation, placement=get_current_placement(), inputs=[lhs, rhs],
    )


def load(key):
    return LoadExpression(placement=get_current_placement(), inputs=[], key=key)


def mul(lhs, rhs):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    return BinaryOpExpression(
        op_type=MulOperation, placement=get_current_placement(), inputs=[lhs, rhs]
    )


def run_program(path, args, *inputs):
    assert isinstance(path, str)
    assert isinstance(args, (list, tuple))
    return RunProgramExpression(
        path=path, args=args, placement=get_current_placement(), inputs=inputs,
    )


def save(value, key):
    assert isinstance(value, Expression)
    return SaveExpression(placement=get_current_placement(), inputs=[value], key=key)


def sub(lhs, rhs):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    return BinaryOpExpression(
        op_type=SubOperation, placement=get_current_placement(), inputs=[lhs, rhs]
    )


def function(input_type=None, output_type=None):
    def callable(fn):
        @wraps(fn)
        def wrapper(*inputs, output_placements=None, **kwargs):
            return ApplyFunctionExpression(
                fn=fn,
                placement=get_current_placement(),
                inputs=inputs,
                output_placements=output_placements,
                output_type=output_type,
            )

        return wrapper

    return callable


class Compiler:
    def __init__(self):
        self.operations = []
        self.name_counters = defaultdict(int)
        self.known_operations = {}

    def compile(self, expression: Expression) -> Computation:
        _ = self.visit(expression)
        operations = self.operations
        graph = Graph(nodes={op.name: op for op in operations})
        computation = Computation(graph=graph)
        return computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}{count}"

    def maybe_add_networking(self, expression, destination_device):
        if destination_device not in self.known_operations[expression]:
            source_device = expression.placement.name
            assert source_device != destination_device
            operation_at_source = self.known_operations[expression][source_device]
            rendezvous_key = self.get_fresh_name("rendezvous_key")
            # Get output type from pyfunction if any
            value_type = getattr(expression, "output_type", None)
            serialize_name = self.get_fresh_name("serialize")
            receive_name = self.get_fresh_name("receive")
            deserialize_name = self.get_fresh_name("deserialize")
            serialize_operation = SerializeOperation(
                device_name=source_device,
                name=self.get_fresh_name("serialize_op"),
                inputs={"value": operation_at_source.output},
                output=serialize_name,
                value_type=value_type,
            )
            send_operation = SendOperation(
                device_name=source_device,
                name=self.get_fresh_name("send_op"),
                inputs={"value": serialize_name},
                output=None,
                sender=source_device,
                receiver=destination_device,
                rendezvous_key=rendezvous_key,
            )
            receive_operation = ReceiveOperation(
                device_name=destination_device,
                name=self.get_fresh_name("receive_op"),
                sender=source_device,
                receiver=destination_device,
                rendezvous_key=rendezvous_key,
                inputs={},
                output=receive_name,
            )
            deserialize_operation = DeserializeOperation(
                device_name=destination_device,
                name=self.get_fresh_name("deserialize_op"),
                inputs={"value": receive_name},
                output=deserialize_name,
                value_type=value_type,
            )
            self.operations += [
                serialize_operation,
                send_operation,
                receive_operation,
                deserialize_operation,
            ]
            self.known_operations[expression][
                destination_device
            ] = deserialize_operation
        return self.known_operations[expression][destination_device]

    def visit(self, expression, destination_device=None):
        device = expression.placement.name
        if expression not in self.known_operations:
            visit_fn = getattr(self, f"visit_{type(expression).__name__}")
            operation = visit_fn(expression)
            self.operations += [operation]
            self.known_operations[expression] = {device: operation}
        return self.maybe_add_networking(expression, destination_device or device)

    def visit_BinaryOpExpression(self, expression):
        assert isinstance(expression, BinaryOpExpression)
        device = expression.placement.name
        lhs_expression, rhs_expression = expression.inputs
        lhs_operation = self.visit(lhs_expression, device)
        rhs_operation = self.visit(rhs_expression, device)
        op_type = expression.op_type
        op_name = op_type.__name__.lower()
        return op_type(
            device_name=device,
            name=self.get_fresh_name(f"{op_name}_op"),
            inputs={"lhs": lhs_operation.output, "rhs": rhs_operation.output},
            output=self.get_fresh_name(f"{op_name}"),
        )

    def visit_ApplyFunctionExpression(self, expression):
        assert isinstance(expression, ApplyFunctionExpression)
        placement = expression.placement
        inputs = {
            f"arg{i}": self.visit(expr, placement.name).output
            for i, expr in enumerate(expression.inputs)
        }
        output_type = expression.output_type
        return placement.compile(
            context=self,
            fn=expression.fn,
            inputs=inputs,
            output_placements=expression.output_placements,
            output_type=output_type,
        )

    def visit_ConstantExpression(self, constant_expression):
        assert isinstance(constant_expression, ConstantExpression)
        return ConstantOperation(
            device_name=constant_expression.placement.name,
            name=self.get_fresh_name("constant_op"),
            value=constant_expression.value,
            inputs={},
            output=self.get_fresh_name("constant"),
        )

    def visit_LoadExpression(self, load_expression):
        assert isinstance(load_expression, LoadExpression)
        return LoadOperation(
            device_name=load_expression.placement.name,
            name=self.get_fresh_name("load_op"),
            key=load_expression.key,
            inputs={},
            output=self.get_fresh_name("load"),
        )

    def visit_RunProgramExpression(self, expression):
        assert isinstance(expression, RunProgramExpression)
        device = expression.placement.name
        inputs = {
            f"arg{i}": self.visit(expr, device).output
            for i, expr in enumerate(expression.inputs)
        }
        return RunProgramOperation(
            device_name=expression.placement.name,
            name=self.get_fresh_name("run_program_op"),
            path=expression.path,
            args=expression.args,
            inputs=inputs,
            output=self.get_fresh_name("run_program"),
        )

    def visit_SaveExpression(self, save_expression):
        assert isinstance(save_expression, SaveExpression)
        save_device = save_expression.placement.name
        (value_expression,) = save_expression.inputs
        value_operation = self.visit(value_expression, save_device)
        return SaveOperation(
            device_name=save_device,
            name=self.get_fresh_name("save_op"),
            key=save_expression.key,
            inputs={"value": value_operation.output},
            output=None,
        )


class AbstractComputation:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        comp = self.trace_func(*args, **kwargs)
        get_runtime().evaluate_computation(comp)

    def trace_func(self, *args, **kwargs):
        expression = self.func(*args, **kwargs)
        compiler = Compiler()
        return compiler.compile(expression)


def computation(func):
    return AbstractComputation(func)
