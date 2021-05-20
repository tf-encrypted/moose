import abc
import functools
import inspect
from typing import Generic, Iterable
from typing import List
from typing import NewType
from typing import Optional
from typing import TypeVar
from typing import Tuple

import numpy as np

from moose import edsl
from moose.compiler.compiler import Compiler
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import StringType
from moose.edsl.base import AbstractComputation
from moose.testing import TestRuntime

class Result:
  def __init__(self, placement, name: Optional[str] = None):
    self.placement = placement
    self.name = name


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

mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
z = mul.run(x=np.array([1.0]), y=np.array([2.0]))
print(z)  # np.array([3.0])

mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
mul.seed_storage("alice", {"x": np.array([1.0])})
mul.seed_storage("bob", {"y": np.array([2.0])})
mul.run(x_key="x", y_key="y", out_key="result")
print(mul.get_results())
# result: np.array([3.0])

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

## Option 1a: decorator-driven
# @place_outputs(output_placements=[(bob, "bobs_x_result"), alice, [alice, bob]])
@edsl.computation
def decorated(
  x: edsl.Argument(alice, dtype=edsl.float32),
  y: edsl.Argument(bob, dtype=edsl.float32),
):
  with multiply_placement:
    z = edsl.mul(x, y)
  return x, y, (z, z)

## Option 1b: signature-driven
# @place_outputs
# @place_computation
@edsl.computation
def other(
  x: edsl.Argument(alice, dtype=edsl.float32),
  y: edsl.Argument(bob, dtype=edsl.float32),
) -> tuple(
    (
      edsl.Result(bob, name="bobs_x_result"),
      edsl.Result(alice),
      (edsl.Result(alice), edsl.Result(bob)),
    )
):
  with multiply_placement:
    z = edsl.mul(x, y)
  return x, y, z

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


def place_computation(abstract_computation=None, output_placements=None):
  if abstract_computation is None and output_placements is None:
    raise ValueError
  elif abstract_computation is None and output_placements is not None:
    return functools.partial(place_computation, output_placements=output_placements)

  edsl_func = abstract_computation.func
  edsl_sig = inspect.signature(edsl_func)
  parameters = [p for p in edsl_sig.parameters.values()]
  if edsl_sig.return_annotation != inspect.Signature.empty:
    updated_annotation = []
    for i, result in enumerate(edsl_sig.return_annotation):
      if isinstance(result, Result):
        if result.name is None:
          updated_annotation.append(Result(result.placement, f"output{i}"))
      elif isinstance(result, (tuple, list)):
        sub_annotation = []
        for j, r in enumerate(result):
          sub_annotation.append(Result(r.placement, f"output{i}_{j}"))
        updated_annotation.append(sub_annotation)
    edsl_sig = edsl_sig.replace(return_annotation=updated_annotation)
  elif output_placements is not None:
    pass
  else:
    raise ValueError

  @edsl.computation
  @functools.wraps(edsl_func)
  def computation_with_storage(*args):  # TODO give these args proper edsl.Argument type hints (if it matters)
    input_values = []
    breakpoint()
    for i, arg in enumerate(args):
      arg_dtype = parameters[i].dtype
      arg_placement = parameters[i].placement
      input_values.append(edsl.load(arg, dtype=arg_dtype, placement=arg_placement))
    return_values = edsl_func(*input_values)
    output_ops = []
    for a, v in zip(edsl_sig.return_annotation, return_values):
      if isinstance(a, edsl.Result):
        op = edsl.save(key=a.name, value=v, placement=a.placement)
      elif isinstance(a, (tuple, list)):
        op = [edsl.save(ai.name, v, placement=ai.placement) for ai in a]
      output_ops.append(op)

    return output_ops

  return computation_with_storage

@place_computation
@edsl.computation
def other(
  x: edsl.Argument(alice, dtype=edsl.float32),
  y: edsl.Argument(bob, dtype=edsl.float32),
) -> tuple(
    (
      edsl.Result(bob, name="bobs_x_result"),
      edsl.Result(alice),
      (edsl.Result(alice), edsl.Result(bob)),
    )
):
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
  

  ### Option 1a ###
  # @place_outputs(placements=[(bob, "bobs_result"), alice, [alice, bob]])
  @edsl.computation
  def other(
    x: edsl.Argument(alice, dtype=edsl.float32),
    y: edsl.Argument(bob, dtype=edsl.float32),
  ):
    with multiply_placement:
      z = edsl.mul(x, y)
    return x, y, (z, z)

  # @place_outputs
  @edsl.computation
  def other(
    x: edsl.Argument(alice, dtype=edsl.float32),
    y: edsl.Argument(bob, dtype=edsl.float32),
  ) -> Tuple[
    edsl.Result(bob, named="bobs_x_result"),
    edsl.Result(alice),
    (edsl.Result(alice), edsl.Result(bob)),
  ]:
    with multiply_placement:
      z = edsl.mul(x, y)
    return x, y, (z, z)

  @edsl.computation
  def simple_multiplication(
      x_key: edsl.Argument(alice, vtype=StringType),
      y_key: edsl.Argument(bob, vtype=StringType),
  ):
    x = edsl.load(x_key, dtype=edsl.float32, placement=alice)
    y = edsl.load(y_key, dtype=edsl.float32, placement=bob)

    with multiply_placement:
      z = edsl.mul(x, y)
    
    with bob:
      res_x = edsl.save("output0", x)
      res_z = edsl.save("output2_1", z)
    with alice:
      res_y = edsl.save("output1", y)
      res_z_alice = edsl.save("output2_0", z)

    return res_x, res_y, res_z, res_z_alice

  return other, simple_multiplication

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
      self._local_runtime = TestRuntime(networking=networking, backing_executors=executor_dict)
    
  def _pretty_print_results(self):
    # need to iterate through executors and pretty print their MemoryDataStore contents
    # if wanted to get fancy could only print stuff that wasn't seeded before evaluation
    # i.e., do something with:
    # {plc: ex.storage.store for plc, ex in self._local_runtime.existing_executors.items()}

    # alternatively, moose could support directly reporting outputs to this python process
    pass

  def local_dry_run(self, *, verbose=False, **computation_kwargs):
    compiler_flags = {
      "passes": None,
      "render": verbose,
    }
    computation = self._compiler.compile(self._logical_comp, **compiler_flags)
    self._build_local_runtime(False)
    plc_instantiation = {plc: plc.name for plc in self.placements}
    # evaluate computation
    self._local_runtime.evaluate_computation(
      computation,
      placement_instantiation=plc_instantiation,
      arguments=computation_kwargs,
    )
    # right now there is no clean way to return results, since we don't know which
    # placements should contain saved values.

    self._pretty_print_results()
    return self  # yes/no??


class MooseEdslMixin(MooseMixin):
  @classmethod
  def from_edsl_func(cls, func):
    if not isinstance(func, AbstractComputation):
      raise ValueError(f"wat? {type(func)}, not: AbstractComputation")
    # it sucks that we need to expose compiler config here; 
    # we should be tracing here and compiling later
    logical_comp = edsl.trace(func)
    return cls.from_computation_object(logical_comp)


class MultiplicationTask(MooseEdslMixin, GraphQLMixin):
  def __init__(self, x_input: DataView, y_input: DataView, secret=False):
    x_placement_name = super(GraphQLMixin).get_placement_name_from_organization(x_input.owner)
    y_placement_name = super(GraphQLMixin).get_placement_name_from_organization(y_input.owner)
    mult_edsl_func = build_multiplication(
      secret=secret,
      x_placement_name=x_placement_name,
      y_placement_name=y_placement_name,
    )
    super().from_edsl_func(mult_edsl_func)


mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
z = mul.run(x=np.array([1.0]), y=np.array([2.0]))
print(z)  # np.array([3.0])

mul = MultiplicationTask("alice", "bob", secret=False)  # x * y = z
mul.seed_storage("alice", {"x": np.array([1.0])})
mul.seed_storage("bob", {"y": np.array([2.0])})
mul.run(x_key="x", y_key="y", out_key="result")
print(mul.get_results())
# result: np.array([3.0])

# WAY IN THE FUTURE STUFF
class JaxTask(MooseMixin):
  @classmethod
  def from_jax_comp(cls, comp): pass  # use jax tracer to produce a fully-functional MooseTask
