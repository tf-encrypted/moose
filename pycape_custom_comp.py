import abc
import functools
import inspect
import itertools
import logging
from typing import Generic, Iterable, TypeVar
from typing import List
from typing import NewType
from typing import Optional
from typing import Tuple

import numpy as np

from moose import edsl
from moose.compiler.compiler import Compiler
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import StringType
from moose.edsl.base import AbstractComputation
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.memory import Networking
from moose.storage.memory import MemoryDataStore
from moose.testing import TestRuntime


def Result(placement, name=None):
    return type("Result", tuple(), {"placement": placement, "name": name})


setattr(edsl, "Result", Result)

##### GOALS #####
# 1. Enable more than 1 computation
# 2. Enable custom Moose computations for quick Cape dev experimentation
# 3. Enable local dry runs on PyCape to catch compiler or user input errors before job submission
# 4. Enable workers to compile computations themselves,
#    i.e. submit job expects traced & uncompiled Moose computation)
# 5. Satisfy these goals while providing a good foundation for future extensions & product features
# 6. Clear path from custom computation -> named/verified/static computation

##### NOTES #####

# mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
# z = mul.run(x=np.array([1.0]), y=np.array([2.0]))
# print(z)  # np.array([3.0])

# mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
# mul.seed_storage("alice", {"x": np.array([1.0])})
# mul.seed_storage("bob", {"y": np.array([2.0])})
# mul.run(x_key="x", y_key="y", out_key="result")
# print(mul.get_results())
# # result: np.array([3.0])

###### EDSL FUNCTION DESIGN OPTIONS ######
# All of these options refer to equivalent computations
# (see Option 2 for the traditional edsl computation)
secret = False
x_placement_name = "alice"
y_placement_name = "bob"
alice = edsl.host_placement(name=x_placement_name)
bob = edsl.host_placement(name=y_placement_name)
if secret:
    cape = edsl.host_placement(name="cape-worker")
    repl = edsl.replicated_placement((alice, bob, cape))
    multiply_placement = repl
else:
    multiply_placement = alice

# Option 1a: decorator-driven
# @place_outputs(output_placements=[(bob, "bobs_x_result"), alice, [alice, bob]])
@edsl.computation
def decorated(
    x: edsl.Argument(alice, dtype=edsl.float32),
    y: edsl.Argument(bob, dtype=edsl.float32),
):
    with multiply_placement:
        z = edsl.mul(x, y)
    return x, y, (z, z)


# # Option 1b: signature-driven
# # @place_outputs
# # @place_computation
# @edsl.computation
# def other(
#     x: edsl.Argument(alice, dtype=edsl.float32),
#     y: edsl.Argument(bob, dtype=edsl.float32),
# ) -> Tuple[
#     edsl.Result(bob, name="bobs_x_result"),
#     edsl.Result(alice),
#     Tuple[edsl.Result(alice), edsl.Result(bob)],
# ]:
#     with multiply_placement:
#         z = edsl.mul(x, y)
#     return x, y, z


# Option 2: no lifting, complete edsl computation
@edsl.computation
def completed(
    x_key: edsl.Argument(alice, vtype=StringType),
    y_key: edsl.Argument(bob, vtype=StringType),
):
    x = edsl.load(x_key, dtype=edsl.float32, placement=alice)
    y = edsl.load(y_key, dtype=edsl.float32, placement=bob)

    with multiply_placement:
        z = edsl.mul(x, y)

    with bob:
        res_x = edsl.save("bobs_x_result", x)
        res_z = edsl.save("output2_1", z)
    with alice:
        res_y = edsl.save("output1", y)
        res_z_alice = edsl.save("output2_0", z)

    return res_x, res_y, res_z, res_z_alice


def _check_wellformed_result(r):
    return hasattr(r, "placement") and hasattr(r, "name")


def _unpack_annotations(annotation, output_name="output"):
    if _check_wellformed_result(annotation):
        if annotation.name is None:
            res = Result(placement=annotation.placement, name=f"{output_name}")
        else:
            res = annotation
        return res
    elif annotation._name == "Tuple":
        return [
            _unpack_annotations(r, output_name=f"{output_name}_{j}")
            for j, r in enumerate(annotation.__args__)
        ]
    else:
        raise ValueError(
            "Return type of edsl function is not nested Tuples of Result types."
        )


def _append_save_ops(annotations, expressions):
    if _check_wellformed_result(annotations):
        return edsl.save(annotations.name, expressions, annotations.placement)
    elif isinstance(annotations, list):
        if isinstance(expressions, (tuple, list)):
            generator = zip(annotations, expressions)
        else:
            generator = zip(annotations, itertools.repeat(expressions))
        return tuple(
            _append_save_ops(annotation, expression)
            for annotation, expression in generator
        )


def _flatten_outputs(outputs):
    for op in outputs:
        if isinstance(op, (tuple, list)):
            for elt in _flatten_outputs(op):
                yield elt
        else:
            yield op


def place_computation(abstract_computation=None, output_placements=None):
    # TODO output_placements?
    if abstract_computation is None and output_placements is None:
        raise ValueError
    elif abstract_computation is None and output_placements is not None:
        return functools.partial(place_computation, output_placements=output_placements)

    edsl_func = abstract_computation.func
    edsl_sig = inspect.signature(edsl_func)
    # capture the original function arg annotations by closure so that
    # we don't have to redefine them below.
    # we use these to get placement & vtype/dtype info for inputs
    parameters = [p for p in edsl_sig.parameters.values()]
    # recursively unpack return type annotations:
    # name any unnamed `Result`s with a numbered naming scheme (e.g. output_1_2).
    # we use return annotations to get placement & naming info for (saved) outputs
    if edsl_sig.return_annotation != inspect.Signature.empty:
        updated_annotation = _unpack_annotations(edsl_sig.return_annotation)
        edsl_sig = edsl_sig.replace(return_annotation=updated_annotation)
    elif output_placements is not None:
        pass
    else:
        raise ValueError

    @edsl.computation
    @functools.wraps(edsl_func)
    def placed_computation(
        *args,
    ):
        # replace input arguments with load operations
        input_expressions = []
        for i, arg in enumerate(args):
            arg_dtype = parameters[i].annotation.dtype
            arg_placement = parameters[i].annotation.placement
            input_expressions.append(
                edsl.load(arg, dtype=arg_dtype, placement=arg_placement)
            )
        return_expressions = edsl_func(*input_expressions)
        # append function return values with save operations
        save_ops = _append_save_ops(edsl_sig.return_annotation, return_expressions)
        breakpoint()
        # edsl tracer expects flattened outputs
        output_ops = [op for op in _flatten_outputs(save_ops)]
        return output_ops

    # replace the function's input annotations with `key` args for the new load ops
    sign = inspect.signature(edsl_func)
    new_params = []
    input_mapping = {}
    for _, param in sign.parameters.items():
        load_name = param.name
        placement = param.annotation.placement
        vtype = StringType
        new_params.append(
            param.replace(
                name=load_name,
                annotation=edsl.Argument(placement=placement, vtype=vtype),
            )
        )
        input_mapping[load_name] = placement

    output_mapping = {}
    for result in _flatten_outputs(edsl_sig.return_annotation):
        if result.placement not in output_mapping:
            output_mapping[result.placement] = [result.name]
        else:
            output_mapping[result.placement].append(result.name)

    new_signature = sign.replace(parameters=new_params)
    setattr(placed_computation.func, "__signature__", new_signature)
    setattr(placed_computation, "_inputs_map", input_mapping)
    setattr(placed_computation, "_outputs_map", output_mapping)
    return placed_computation


@place_computation
@edsl.computation
def other(
    x: edsl.Argument(alice, dtype=edsl.float32),
    y: edsl.Argument(bob, dtype=edsl.float32),
) -> Tuple[
    edsl.Result(bob),
    edsl.Result(alice),
    Tuple[edsl.Result(alice), edsl.Result(bob)],
]:
    with multiply_placement:
        z = edsl.mul(x, y)
    return x, y, z


##### EXAMPLE TASK EXPECTED FROM PYCAPE DEV AND/OR USER ######
def build_multiplication(secret, x_placement_name, y_placement_name):
    alice = edsl.host_placement(name=x_placement_name)
    bob = edsl.host_placement(name=y_placement_name)
    if secret:
        cape = edsl.host_placement(name="cape-worker")
        repl = edsl.replicated_placement((alice, bob, cape))
        multiply_placement = repl
    else:
        multiply_placement = alice

    @place_computation
    @edsl.computation
    def other(
        x: edsl.Argument(alice, dtype=edsl.float32),
        y: edsl.Argument(bob, dtype=edsl.float32),
    ) -> Tuple[
        edsl.Result(bob),
        edsl.Result(alice),
        Tuple[edsl.Result(alice), edsl.Result(bob)],
    ]:
        with multiply_placement:
            z = edsl.mul(x, y)
        return x, y, z

    return other


##############################################################


# ###### SCRATCH USAGE ######
# task = MooseEdslTask.from_edsl_func(simple_multiplication, render=True)
# # or equivalently
# task = task.compile(compiler_config=None)  # compiler flags instead?
# computation_kwargs = {"x": np.array([1.0]), "y": np.array([2.0])}
# task = task.dry_run(verbose=False, **computation_kwargs)
# # verbose=True pretty prints results of local dry run (+ filepaths to optional graph renders)
# task.get_placement_store(placement_name)
# project.create_task(task)
# ###########################


# can make this less transparent if we want (e.g. make it collection of compiler
# flags)
# right now it's pretty tied into the Moose compiler interface, which is likely to
# change in the near future
class CompilerConfig:
    def __init__(self, compiler_passes=None, render=False):
        self.passes = compiler_passes
        self.render = render


class GraphQLMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def lift_to_graphql_request(self, **kwargs):  # TODO: function sig?
        pass

    def submit_job(self, **kwargs):  # TODO: function sig?
        self.lift_to_graphql_request(**kwargs)


class MooseMixin:
    def __init__(self, comp, compiler_config=None):
        self._logical_comp = comp
        self._compiler_config = compiler_config or CompilerConfig()
        self._compiler = Compiler()
        self._networking = None
        self._storage_dict = None
        self._executor_dict = None
        self._local_runtime = None

    @classmethod
    def from_computation_object(cls, comp):
        if not isinstance(comp, Computation):
            raise ValueError(f"wat? {type(comp)}, not: Computation")
        return cls(comp)

    def _find_host_placements(self, comp):
        host_placements = dict()
        for name, placement in comp.placements.items():
            if isinstance(placement, HostPlacement):
                host_placements[name] = placement
            elif hasattr(placement, "player_names"):
                # replicated or mpspdz placement
                for name in placement.player_names:
                    player_placement = comp.placement(name)
                    assert isinstance(player_placement, HostPlacement)
                    host_placements.update({name: player_placement})
        return host_placements

    @property
    def placements(self):
        if self._placements is None:
            self._placements = self._find_host_placements(self._logical_comp)
        return self._placements

    def seed_storage(self, placement, *, **keyvalue_store):
        # may need to provide initial storage values for some placements
        assert self._local_runtime is not None, "Must build local runtime."

    def _build_local_runtime(self, force_rebuild=False):
        if self._local_runtime is None or force_rebuild:
            networking = Networking()
            # Storage for each executor
            storage_dict = {plc.name: MemoryDataStore() for plc in self.placements}
            # Executors
            executor_dict = {
                plc.name: AsyncExecutor(networking, storage=storage_dict[plc.name])
                for plc in self.placements
            }
            # Runtime
            self._local_runtime = TestRuntime(
                networking=networking, backing_executors=executor_dict
            )

    def _pretty_print_results(self):
        # need to iterate through executors and pretty print their MemoryDataStore contents
        # if wanted to get fancy could only print stuff that wasn't seeded before evaluation
        # i.e., do something with:
        # {plc: ex.storage.store for plc, ex in self._local_runtime.existing_executors.items()}

        # alternatively, moose could support directly reporting outputs to this python process
        pass

    def run(self, *, verbose=False, render=False, **computation_kwargs):
        if verbose:
            get_logger().setLevel(level=logging.DEBUG)
        compiler_flags = {
            "passes": None,
            "render": render,
        }
        computation = self._compiler.compile(self._logical_comp, **compiler_flags)
        self._build_local_runtime(False)
        # the edsl func we've traced (via @place_computation) has all the info we need
        # to seed storage for inputs and also extract computation results from storage.
        # however, with MooseEdslMixin vs MooseMixin, this information would get erased,
        # so maybe we need to pull from the Computation object itself
        # self.seed_storage(sateohusaoetu)
        plc_instantiation = {plc: plc.name for plc in self.placements}
        # evaluate computation
        self._local_runtime.evaluate_computation(
            computation,
            placement_instantiation=plc_instantiation,
            arguments=computation_kwargs,
        )
        # self.pull_results(asoetuhasoe)


class MooseEdslMixin(MooseMixin):
    @classmethod
    def from_edsl_func(cls, func):
        if not isinstance(func, AbstractComputation):
            raise ValueError(f"wat? {type(func)}, not: AbstractComputation")
        # it sucks that we need to expose compiler config here;
        # we should be tracing here and compiling later
        logical_comp = edsl.trace(func)
        return cls.from_computation_object(logical_comp)

class MultiplicationTask(MooseMixin, GraphQLMixin):
    def create(cls):
        comp = Computation()
        comp.add_operation(staoheu)
        comp.add_operation(MulOperation)
        comp.add_operation(staoheu)
        comp.add_operation(staoheu)

        return super(cls, MooseMixin).from_computation_object(comp)


class MultiplicationTask(MooseEdslMixin, GraphQLMixin):
    def __init__(self, x_input, y_input, secret=False):
        x_placement_name = super(GraphQLMixin).get_placement_name_from_organization(
            x_input.owner
        )
        y_placement_name = super(GraphQLMixin).get_placement_name_from_organization(
            y_input.owner
        )
        mult_edsl_func = build_multiplication(
            secret=secret,
            x_placement_name=x_placement_name,
            y_placement_name=y_placement_name,
        )
        super().from_edsl_func(mult_edsl_func)


# mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
# z = mul.run(x=np.array([1.0]), y=np.array([2.0]))
# print(z)  # np.array([3.0])

# mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
# mul.seed_storage("alice", {"x": np.array([1.0])})
# mul.seed_storage("bob", {"y": np.array([2.0])})
# mul.run(x_key="x", y_key="y", out_key="result")
# print(mul.get_results())
# # result: np.array([3.0])


# WAY IN THE FUTURE STUFF
class JaxTask(MooseMixin):
    @classmethod
    def from_jax_comp(cls, comp):
        pass  # use jax tracer to produce a fully-functional MooseTask

def test_local_computation():
    comp_inputs = {
        "x": np.array([2.0]),
        "y": np.array([3.0])
    }

    @place_computation
    @edsl.computation
    def other(
        x: edsl.Argument(alice, dtype=edsl.float32),
        y: edsl.Argument(bob, dtype=edsl.float32),
    ) -> Tuple[
        edsl.Result(bob), edsl.Result(alice), Tuple[edsl.Result(alice), edsl.Result(bob)],
    ]:
        with multiply_placement:
            z = edsl.mul(x, y)
        return x, y, z


    local_computation = edsl.trace_and_compile(other)
    networking = Networking()
    # Storage for each executor
    placements = local_computation.placements.values()
    storage_dict = {plc.name: MemoryDataStore() for plc in placements}
    executor_dict = {
        plc.name: AsyncExecutor(networking, storage=storage_dict[plc.name])
        for plc in placements
    }
    # Runtime
    local_runtime = TestRuntime(
        networking=networking, backing_executors=executor_dict
    )
    # placement instantiation
    plc_inst = {plc: plc.name for plc in placements}
    local_runtime.evaluate_computation(local_computation, plc_inst, comp_inputs)


if __name__ == "__main__":
    verbose = True
    if verbose:
        get_logger().setLevel(level=logging.DEBUG)
    test_local_computation()
