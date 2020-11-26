from collections import defaultdict
from dataclasses import dataclass
from functools import partial
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
        inputs = {f"arg{i}": context.visit(expr).name for i, expr in enumerate(inputs)}
        return CallPythonFunctionOperation(
            placement_name=self.name,
            name=context.get_fresh_name("call_python_function"),
            pickled_fn=dill.dumps(fn),
            inputs=inputs,
            output_type=output_type,
        )


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


def add(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(
        op_type=AddOperation, placement=placement, inputs=[lhs, rhs]
    )


def constant(value, placement=None):
    placement = placement or get_current_placement()
    return ConstantExpression(placement=placement, inputs=[], value=value)


def div(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(
        op_type=DivOperation, placement=placement, inputs=[lhs, rhs]
    )


def load(key, placement=None):
    placement = placement or get_current_placement()
    return LoadExpression(placement=placement, inputs=[], key=key)


def mul(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(
        op_type=MulOperation, placement=placement, inputs=[lhs, rhs]
    )


def run_program(path, args, *inputs, placement=None):
    assert isinstance(path, str)
    assert isinstance(args, (list, tuple))
    placement = placement or get_current_placement()
    return RunProgramExpression(
        path=path, args=args, placement=placement, inputs=inputs
    )


def save(value, key, placement=None):
    assert isinstance(value, Expression)
    placement = placement or get_current_placement()
    return SaveExpression(placement=placement, inputs=[value], key=key)


def sub(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(
        op_type=SubOperation, placement=placement, inputs=[lhs, rhs]
    )


def function(fn=None, output_type=None):
    if fn is None:
        return partial(function, output_type=output_type)

    @wraps(fn)
    def wrapper(*inputs, placement=None, output_placements=None, **kwargs):
        return ApplyFunctionExpression(
            fn=fn,
            placement=placement or get_current_placement(),
            inputs=inputs,
            output_placements=output_placements,
            output_type=output_type,
        )

    return wrapper


class NetworkingPass:
    def __init__(self, reuse_when_possible=True):
        self.reuse_when_possible = reuse_when_possible
        self.serialize_cache = dict()
        self.deserialize_cache = dict()

    def process(self, computation, context):
        # we first find all edges to cut since we cannot mutate dict while traversing
        # TODO(Morten) this could probably be improved
        edges_to_cut = []
        for dst_op in computation.operations():
            for input_key, input_name in dst_op.inputs.items():
                src_op = computation.operation(input_name)
                if src_op.placement_name != dst_op.placement_name:
                    edges_to_cut += [(src_op, dst_op, input_key)]

        # cut each edge and replace with networking ops
        # we keep a cache of certain ops to avoid redundancy
        for src_op, dst_op, input_key in edges_to_cut:
            patched_src_op, extra_ops = self.add_networking(
                context, src_op, dst_op.placement_name
            )
            computation.add_operations(extra_ops)
            dst_op.inputs[input_key] = patched_src_op.name

        return computation

    def add_networking(self, context, source_operation, destination_placement_name):
        extra_ops = []

        if source_operation.placement_name == destination_placement_name:
            # nothing to do, we are already on the same placement
            return source_operation, extra_ops

        derialize_cache_key = (source_operation.name, destination_placement_name)
        if self.reuse_when_possible and derialize_cache_key in self.deserialize_cache:
            # nothing do do, we can reuse everything
            return self.deserialize_cache[derialize_cache_key], extra_ops

        # maybe we can reuse the serialized value
        serialize_cache_key = (source_operation.name, source_operation.placement_name)
        if self.reuse_when_possible and serialize_cache_key in self.serialize_cache:
            serialize_operation = self.serialize_cache[serialize_cache_key]
        else:
            serialize_operation = SerializeOperation(
                placement_name=source_operation.placement_name,
                name=context.get_fresh_name("serialize"),
                inputs={"value": source_operation.name},
                value_type=getattr(source_operation, "output_type", None),
            )
            self.serialize_cache[serialize_cache_key] = serialize_operation
            extra_ops += [serialize_operation]

        rendezvous_key = context.get_fresh_name("rendezvous_key")
        send_operation = SendOperation(
            placement_name=source_operation.placement_name,
            name=context.get_fresh_name("send"),
            inputs={"value": serialize_operation.name},
            sender=source_operation.placement_name,
            receiver=destination_placement_name,
            rendezvous_key=rendezvous_key,
        )
        receive_operation = ReceiveOperation(
            placement_name=destination_placement_name,
            name=context.get_fresh_name("receive"),
            inputs={},
            sender=source_operation.placement_name,
            receiver=destination_placement_name,
            rendezvous_key=rendezvous_key,
        )
        deserialize_operation = DeserializeOperation(
            placement_name=destination_placement_name,
            name=context.get_fresh_name("deserialize"),
            inputs={"value": receive_operation.name},
            value_type=serialize_operation.value_type,
        )
        self.deserialize_cache[derialize_cache_key] = deserialize_operation
        extra_ops += [send_operation, receive_operation, deserialize_operation]
        return deserialize_operation, extra_ops


class Compiler:
    def __init__(self, passes=None):
        self.passes = passes or [NetworkingPass()]
        self.operations = []
        self.name_counters = defaultdict(int)
        self.known_operations = defaultdict(dict)

    def compile(self, expression: Expression, render=False) -> Computation:
        _ = self.visit(expression)
        graph = Graph(nodes={op.name: op for op in self.operations})
        computation = Computation(graph=graph)
        if render:
            computation.render("Logical")
        for compiler_pass in self.passes:
            computation = compiler_pass.process(computation, context=self)
            if render:
                computation.render(f"{type(compiler_pass).__name__}")
        return computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"

    def visit(self, expression):
        if expression not in self.known_operations:
            visit_fn = getattr(self, f"visit_{type(expression).__name__}")
            operation = visit_fn(expression)
            self.operations += [operation]
            self.known_operations[expression] = operation
        return self.known_operations[expression]

    def visit_BinaryOpExpression(self, expression):
        assert isinstance(expression, BinaryOpExpression)
        lhs_expression, rhs_expression = expression.inputs
        lhs_operation = self.visit(lhs_expression)
        rhs_operation = self.visit(rhs_expression)
        op_type = expression.op_type
        op_name = op_type.__name__.lower()
        if op_name.endswith("operation"):
            op_name = op_name[: -len("operation")]
        return op_type(
            placement_name=expression.placement.name,
            name=self.get_fresh_name(f"{op_name}"),
            inputs={"lhs": lhs_operation.name, "rhs": rhs_operation.name},
        )

    def visit_ApplyFunctionExpression(self, expression):
        assert isinstance(expression, ApplyFunctionExpression)
        return expression.placement.compile(
            context=self,
            fn=expression.fn,
            inputs=expression.inputs,
            output_placements=expression.output_placements,
            output_type=expression.output_type,
        )

    def visit_ConstantExpression(self, constant_expression):
        assert isinstance(constant_expression, ConstantExpression)
        return ConstantOperation(
            placement_name=constant_expression.placement.name,
            name=self.get_fresh_name("constant"),
            value=constant_expression.value,
            inputs={},
        )

    def visit_LoadExpression(self, load_expression):
        assert isinstance(load_expression, LoadExpression)
        return LoadOperation(
            placement_name=load_expression.placement.name,
            name=self.get_fresh_name("load"),
            key=load_expression.key,
            inputs={},
        )

    def visit_RunProgramExpression(self, expression):
        assert isinstance(expression, RunProgramExpression)
        inputs = {
            f"arg{i}": self.visit(expr).name for i, expr in enumerate(expression.inputs)
        }
        return RunProgramOperation(
            placement_name=expression.placement.name,
            name=self.get_fresh_name("run_program"),
            path=expression.path,
            args=expression.args,
            inputs=inputs,
        )

    def visit_SaveExpression(self, save_expression):
        assert isinstance(save_expression, SaveExpression)
        (value_expression,) = save_expression.inputs
        value_operation = self.visit(value_expression)
        return SaveOperation(
            placement_name=save_expression.placement.name,
            name=self.get_fresh_name("save"),
            key=save_expression.key,
            inputs={"value": value_operation.name},
        )


class AbstractComputation:
    def __init__(self, func):
        self.func = func

    # TODO(Morten) we could bring this back later for eg RemoteRuntime only
    # def __call__(self, *args, **kwargs):
    #     comp = self.trace_func(*args, **kwargs)
    #     get_runtime().evaluate_computation(comp)

    def trace_func(self, *args, **kwargs):
        expression = self.func(*args, **kwargs)
        compiler = Compiler()
        concrete_comp = compiler.compile(expression)
        for op in concrete_comp.operations():
            get_logger().debug(f"Computation: {op}")
        return concrete_comp


def computation(func):
    return AbstractComputation(func)
