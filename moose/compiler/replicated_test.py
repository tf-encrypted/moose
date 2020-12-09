from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.compiler.replicated import ReplicatedFromStandardOpsPass
from moose.compiler.replicated import ReplicatedShareRevealPass
from moose.computation import replicated as replicated_ops
from moose.computation import standard as standard_ops
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement


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
        comp.add_placement(HostPlacement(name="eric"))

        comp.add_operation(
            standard_ops.ConstantOperation(
                name="alice_input", inputs={}, value=1, placement_name="alice"
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input", inputs={}, value=2, placement_name="bob"
            )
        )

        comp.add_operation(
            standard_ops.AddOperation(
                name="secure_add",
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                placement_name="rep",
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "secure_add", "rhs": "secure_add"},
                placement_name="dave",
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "add_dave"}, placement_name="dave",
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "secure_add", "rhs": "secure_add"},
                placement_name="eric",
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1", inputs={"value": "add_eric"}, placement_name="eric",
            )
        )

        compiler = Compiler(
            passes=[ReplicatedFromStandardOpsPass(), ReplicatedShareRevealPass()]
        )
        comp = compiler.run_passes(comp, render=True)

        expected_comp = Computation(placements={}, operations={})
        expected_comp.add_placement(HostPlacement(name="alice"))
        expected_comp.add_placement(HostPlacement(name="bob"))
        expected_comp.add_placement(HostPlacement(name="carole"))
        expected_comp.add_placement(
            ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])
        )
        expected_comp.add_placement(HostPlacement(name="dave"))
        expected_comp.add_placement(HostPlacement(name="eric"))
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="alice_input", inputs={}, value=1, placement_name="alice"
            )
        )
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input", inputs={}, value=2, placement_name="bob"
            )
        )
        expected_comp.add_operation(
            replicated_ops.SetupOperation(
                name="replicated_setup_0", inputs={}, placement_name="rep",
            )
        )
        expected_comp.add_operation(
            replicated_ops.ShareOperation(
                name="share_0",
                inputs={"setup": "replicated_setup_0", "value": "alice_input"},
                placement_name="rep",
            )
        )
        expected_comp.add_operation(
            replicated_ops.ShareOperation(
                name="share_1",
                inputs={"setup": "replicated_setup_0", "value": "bob_input"},
                placement_name="rep",
            )
        )
        expected_comp.add_operation(
            replicated_ops.AddOperation(
                name="replicated_add_0",
                inputs={
                    "setup": "replicated_setup_0",
                    "lhs": "share_0",
                    "rhs": "share_1",
                },
                placement_name="rep",
            )
        )
        expected_comp.add_operation(
            replicated_ops.RevealOperation(
                name="reveal_0",
                inputs={"setup": "replicated_setup_0", "value": "replicated_add_0"},
                recipient_name="dave",
                placement_name="rep",
            )
        )
        expected_comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "reveal_0", "rhs": "reveal_0"},
                placement_name="dave",
            )
        )
        expected_comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "add_dave"}, placement_name="dave",
            )
        )
        expected_comp.add_operation(
            replicated_ops.RevealOperation(
                name="reveal_1",
                inputs={"setup": "replicated_setup_0", "value": "replicated_add_0"},
                recipient_name="eric",
                placement_name="rep",
            )
        )
        expected_comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "reveal_1", "rhs": "reveal_1"},
                placement_name="eric",
            )
        )
        expected_comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1", inputs={"value": "add_eric"}, placement_name="eric",
            )
        )

        assert comp.placements == expected_comp.placements
        assert comp == expected_comp

    def test_replicated_lowering(self):

        comp = Computation(placements={}, operations={})

        comp.add_placement(HostPlacement(name="alice"))
        comp.add_placement(HostPlacement(name="bob"))
        comp.add_placement(HostPlacement(name="carole"))
        comp.add_placement(
            ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])
        )
        comp.add_placement(HostPlacement(name="dave"))
        comp.add_placement(HostPlacement(name="eric"))

        comp.add_operation(
            standard_ops.ConstantOperation(
                name="alice_input", inputs={}, value=1, placement_name="alice"
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input", inputs={}, value=2, placement_name="bob"
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="secure_add",
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                placement_name="rep",
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "secure_add", "rhs": "secure_add"},
                placement_name="dave",
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "add_dave"}, placement_name="dave",
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "secure_add", "rhs": "secure_add"},
                placement_name="eric",
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1", inputs={"value": "add_eric"}, placement_name="eric",
            )
        )

        compiler = Compiler()
        comp = compiler.run_passes(comp)

        assert all(
            isinstance(comp.placement(op.placement_name), HostPlacement)
            for op in comp.operations.values()
        )
