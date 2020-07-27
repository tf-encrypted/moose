from collections import defaultdict
from dataclasses import dataclass
from typing import List

from computation import AddOperation
from computation import Computation
from computation import Graph
from computation import LoadOperation
from computation import MulOperation
from computation import ReceiveOperation
from computation import SaveOperation
from computation import SendOperation
from computation import SubOperation
from runtime import get_runtime

CURRENT_ROLE: List = []


def get_current_role():
    global CURRENT_ROLE
    return CURRENT_ROLE[-1]


@dataclass
class Role:
    name: str

    def __hash__(self):
        return hash(self.name)

    def __enter__(self):
        global CURRENT_ROLE
        CURRENT_ROLE.append(self)

    def __exit__(self, type, value, traceback):
        global CURRENT_ROLE
        CURRENT_ROLE.pop(-1)


@dataclass
class Expression:
    role: Role
    inputs: List

    def __hash__(self):
        return id(self)


@dataclass
class LoadExpression(Expression):
    key: str

    def __hash__(self):
        return id(self)


@dataclass
class SaveExpression(Expression):
    key: str

    def __hash__(self):
        return id(self)


@dataclass
class AddExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SubExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class MulExpression(Expression):
    def __hash__(self):
        return id(self)


def load(key):
    return LoadExpression(role=get_current_role(), inputs=[], key=key)


def save(value, key):
    assert isinstance(value, Expression)
    return SaveExpression(role=get_current_role(), inputs=[value], key=key)


def add(lhs, rhs):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    return AddExpression(role=get_current_role(), inputs=[lhs, rhs],)


def sub(lhs, rhs):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    return SubExpression(role=get_current_role(), inputs=[lhs, rhs],)


def mul(lhs, rhs):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    return MulExpression(role=get_current_role(), inputs=[lhs, rhs],)


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
        name = "{}{}".format(prefix, count)
        return name

    def maybe_add_networking(self, expression, destination_device):
        if destination_device not in self.known_operations[expression]:
            source_device = expression.role.name
            assert source_device != destination_device
            operation_at_source = self.known_operations[expression][source_device]
            channel_name = "{}_{}".format(source_device, destination_device)
            rendezvous_key = self.get_fresh_name("rendezvous_key")
            send_operation = SendOperation(
                device_name=source_device,
                name=self.get_fresh_name("send_op"),
                inputs={"value": operation_at_source.output},
                output=None,
                channel=channel_name,
                rendezvous_key=rendezvous_key,
            )
            receive_operation = ReceiveOperation(
                device_name=destination_device,
                name=self.get_fresh_name("receive_op"),
                channel=channel_name,
                rendezvous_key=rendezvous_key,
                inputs={},
                output=self.get_fresh_name("receive"),
            )
            self.operations += [send_operation, receive_operation]
            self.known_operations[expression][destination_device] = receive_operation
        return self.known_operations[expression][destination_device]

    def visit(self, expression, destination_device=None):
        device = expression.role.name
        if expression not in self.known_operations:
            visit_fn = getattr(self, "visit_{}".format(type(expression).__name__))
            operation = visit_fn(expression)
            self.operations += [operation]
            self.known_operations[expression] = {device: operation}
        return self.maybe_add_networking(expression, destination_device or device)

    def visit_LoadExpression(self, load_expression):
        assert isinstance(load_expression, LoadExpression)
        return LoadOperation(
            device_name=load_expression.role.name,
            name=self.get_fresh_name("load_op"),
            key=load_expression.key,
            inputs={},
            output=self.get_fresh_name("load"),
        )

    def visit_SaveExpression(self, save_expression):
        assert isinstance(save_expression, SaveExpression)
        save_device = save_expression.role.name
        (value_expression,) = save_expression.inputs
        value_operation = self.visit(value_expression, save_device)
        return SaveOperation(
            device_name=save_device,
            name=self.get_fresh_name("save_op"),
            key=save_expression.key,
            inputs={"value": value_operation.output},
            output=None,
        )

    def visit_AddExpression(self, add_expression):
        assert isinstance(add_expression, AddExpression)
        add_device = add_expression.role.name
        lhs_expression, rhs_expression = add_expression.inputs
        lhs_operation = self.visit(lhs_expression, add_device)
        rhs_operation = self.visit(rhs_expression, add_device)
        return AddOperation(
            device_name=add_device,
            name=self.get_fresh_name("add_op"),
            inputs={"lhs": lhs_operation.output, "rhs": rhs_operation.output},
            output=self.get_fresh_name("add"),
        )

    def visit_SubExpression(self, sub_expression):
        assert isinstance(sub_expression, SubExpression)
        sub_device = sub_expression.role.name
        lhs_expression, rhs_expression = sub_expression.inputs
        lhs_operation = self.visit(lhs_expression, sub_device)
        rhs_operation = self.visit(rhs_expression, sub_device)
        return SubOperation(
            device_name=sub_device,
            name=self.get_fresh_name("sub_op"),
            inputs={"lhs": lhs_operation.output, "rhs": rhs_operation.output},
            output=self.get_fresh_name("sub"),
        )

    def visit_MulExpression(self, mul_expression):
        assert isinstance(mul_expression, MulExpression)
        mul_device = mul_expression.role.name
        lhs_expression, rhs_expression = mul_expression.inputs
        lhs_operation = self.visit(lhs_expression, mul_device)
        rhs_operation = self.visit(rhs_expression, mul_device)
        return MulOperation(
            device_name=mul_device,
            name=self.get_fresh_name("mul_op"),
            inputs={"lhs": lhs_operation.output, "rhs": rhs_operation.output},
            output=self.get_fresh_name("mul"),
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
