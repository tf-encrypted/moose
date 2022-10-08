from typing import Dict
from typing import List
from typing import Optional

from pymoose.computation import computation as comp
from pymoose.computation import utils
from pymoose.edsl import base as edsl
from pymoose.edsl import tracer
from pymoose.edsl.base import HostPlacementExpression
from pymoose.edsl.base import set_current_runtime
from pymoose.pymoose import moose_runtime


class LocalMooseRuntime(moose_runtime.LocalRuntime):
    """Locally-simulated Moose runtime.

    Creates a local runtime with several virtual hosts and optional storage for each.

    Example:
        runtime = LocalMooseRuntime(
            ["alice", "bob", "carole"],
            storage_mapping={
                "alice": {
                    "alice_array": np.ones((3, 3)),
                    "alice_string": "hello i'm alice",
                    "alice_number": 42,
                },
            },
        )

    Args:
        identities: A list of names of identities for virtual hosts.
        storage_mapping: A dictionary mapping Identities to storage dictionaries.
            Storage dictionaries are dicts/named tuples of Python objects that
            correspond to valid PyMoose Values (e.g. native Python numbers,
            ndarrays, strings, etc.).
    """

    def __new__(
        cls, identities: List[str], storage_mapping: Optional[Dict[str, Dict]] = None
    ):
        storage_mapping = storage_mapping or {}
        for identity in storage_mapping:
            if identity not in identities:
                raise ValueError(
                    f"Found unknown identity {identity} in `storage_mapping` arg, "
                    f"must be one of {identities}."
                )
        for identity in identities:
            if identity not in storage_mapping:
                storage_mapping[identity] = {}
        return moose_runtime.LocalRuntime.__new__(
            LocalMooseRuntime, storage_mapping=storage_mapping
        )

    def set_default(self):
        set_current_runtime(self)

    def evaluate_computation(
        self,
        computation,
        arguments=None,
        compiler_passes=None,
    ):
        computation, arguments = _lift_comp_and_args(computation, arguments)
        comp_bin = utils.serialize_computation(computation)
        return super().evaluate_computation(comp_bin, arguments, compiler_passes)

    def evaluate_compiled(self, comp_bin, arguments=None):
        if arguments is None:
            arguments = {}
        return super().evaluate_compiled(comp_bin, arguments)

    def read_value_from_storage(self, identity, key):
        return super().read_value_from_storage(identity, key)

    def write_value_to_storage(self, identity, key, value):
        return super().write_value_to_storage(identity, key, value)


class GrpcMooseRuntime(moose_runtime.GrpcRuntime):
    """Moose runtime backed by gRPC choreography.

    Creates a Moose runtime backed by a fixed set of gRPC servers.

    Args:
        identities: Mapping of identities (e.g. host placement identifiers) to gRPC
            host addresses.
    """

    def __new__(cls, identities: Dict):
        identities = {
            role.name if isinstance(role, HostPlacementExpression) else role: addr
            for role, addr in identities.items()
        }
        return moose_runtime.GrpcRuntime.__new__(GrpcMooseRuntime, identities)

    def set_default(self):
        set_current_runtime(self)

    def evaluate_computation(
        self,
        computation,
        arguments=None,
    ):
        computation, arguments = _lift_comp_and_args(computation, arguments)
        comp_bin = utils.serialize_computation(computation)
        return super().evaluate_computation(comp_bin, arguments)


def _lift_comp_and_args(computation, arguments):
    if not isinstance(computation, (edsl.AbstractComputation, comp.Computation)):
        raise ValueError(
            "`computation` arg must be of type AbstractComputation or Computation, "
            f"found type {type(computation)}."
        )

    if isinstance(computation, edsl.AbstractComputation):
        computation = tracer.trace(computation)

    if arguments is None:
        arguments = {}

    return computation, arguments
