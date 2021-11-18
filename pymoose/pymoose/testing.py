from pymoose import LocalRuntime
from pymoose import edsl


class LocalMooseRuntime(LocalRuntime):
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
        return LocalRuntime.__new__(LocalMooseRuntime, storage_mapping=storage_mapping)

    def evaluate_computation(
        self, computation, role_assignment, arguments=None, ring=128
    ):
        if arguments is None:
            arguments = {}
        concrete_comp_ref = edsl.trace_and_compile(computation)
        comp_outputs = super().evaluate_compiled(
            concrete_comp_ref, role_assignment, arguments
        )
        outputs = list(dict(sorted(comp_outputs.items())).values())
        return outputs

    def evaluate_compiled(self, comp_bin, role_assignment, arguments=None, ring=128):
        if arguments is None:
            arguments = {}
        comp_outputs = super().evaluate_compiled(comp_bin, role_assignment, arguments)
        outputs = list(dict(sorted(comp_outputs.items())).values())
        return outputs

    def read_value_from_storage(self, identity, key):
        return super().read_value_from_storage(identity, key)

    def write_value_to_storage(self, identity, key, value):
        return super().write_value_to_storage(identity, key, value)
