from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.replicated import RevealOperation
from moose.computation.replicated import ShareOperation
from moose.computation.standard import AddOperation
from moose.computation.standard import ConstantOperation


class ReplicatedTest(parameterized.TestCase):
    def test_replicated(self):

        comp = Computation(placements={}, operations={})

        comp.add_placement(HostPlacement(name="alice"))
        comp.add_placement(HostPlacement(name="bob"))
        comp.add_placement(HostPlacement(name="carole"))
        comp.add_placement(
            ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])
        )
        comp.add_placement(HostPlacement(name="dave"))

        comp.add_operation(
            ConstantOperation(
                name="alice_input", inputs={}, value=1, placement_name="alice"
            )
        )
        comp.add_operation(
            ConstantOperation(
                name="bob_input", inputs={}, value=2, placement_name="bob"
            )
        )
        comp.add_operation(
            ShareOperation(
                name="share_alice_input",
                inputs={"value": "alice_input"},
                placement_name="rep",
            )
        )
        comp.add_operation(
            ShareOperation(
                name="share_bob_input",
                inputs={"value": "bob_input"},
                placement_name="rep",
            )
        )
        comp.add_operation(
            AddOperation(
                name="secure_add",
                inputs={"lhs": "share_alice_input", "rhs": "share_bob_input"},
                placement_name="rep",
            )
        )
        comp.add_operation(
            RevealOperation(
                name="reveal_output",
                inputs={"value": "secure_add"},
                recipient_name="dave",
                placement_name="rep",
            )
        )

        compiler = Compiler()
        comp = compiler.run_passes(comp, render=True)

        assert all(
            isinstance(comp.placement(op.placement_name), HostPlacement)
            for op in comp.operations.values()
        )
