from pymoose.computation import utils
from pymoose.edsl.base import AbstractComputation
from pymoose.edsl.tracer import trace
from pymoose.rust import moose_runtime


class LocalMooseRuntime(moose_runtime.LocalRuntime):
    def __new__(cls, *, identities=None, storage_mapping=None):
        if identities is None and storage_mapping is None:
            raise ValueError(
                "Must provide either a list of identities or a mapping of identities "
                "to executor storage dicts."
            )
        elif storage_mapping is not None and identities is not None:
            assert storage_mapping.keys() == identities
        elif identities is not None:
            storage_mapping = {identity: {} for identity in identities}
        return moose_runtime.LocalRuntime.__new__(
            LocalMooseRuntime, storage_mapping=storage_mapping
        )

    def evaluate_computation(
        self,
        computation,
        role_assignment,
        arguments=None,
        compiler_passes=None,
    ):
        if arguments is None:
            arguments = {}
        comp_bin = utils.serialize_computation(computation)
        return super().evaluate_computation(
            comp_bin, role_assignment, arguments, compiler_passes
        )

    def evaluate_compiled(self, comp_bin, role_assignment, arguments=None):
        if arguments is None:
            arguments = {}
        return super().evaluate_compiled(comp_bin, role_assignment, arguments)

    def read_value_from_storage(self, identity, key):
        return super().read_value_from_storage(identity, key)

    def write_value_to_storage(self, identity, key, value):
        return super().write_value_to_storage(identity, key, value)


class GrpcMooseRuntime(moose_runtime.GrpcRuntime):
    def __new__(cls, role_assignment):
        return moose_runtime.GrpcRuntime.__new__(GrpcMooseRuntime, role_assignment)

    def evaluate_computation(
        self,
        computation,
        arguments=None,
    ):
        if isinstance(computation, AbstractComputation):
            computation = trace(computation)

        if arguments is None:
            arguments = {}

        comp_bin = utils.serialize_computation(computation)
        return super().evaluate_computation(comp_bin, arguments)
