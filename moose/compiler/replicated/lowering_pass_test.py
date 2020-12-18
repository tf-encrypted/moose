import unittest

from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.compiler.replicated.encoding_pass import ReplicatedEncodingPass
from moose.compiler.replicated.lowering_pass import ReplicatedLoweringPass
from moose.compiler.replicated.replicated_pass import ReplicatedOpsPass
from moose.computation import standard as standard_ops
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import TensorType


class ReplicatedTest(parameterized.TestCase):
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
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input",
                inputs={},
                value=2,
                placement_name="bob",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add",
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                placement_name="rep",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "add", "rhs": "add"},
                placement_name="dave",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "add_dave"}, placement_name="dave"
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "add", "rhs": "add"},
                placement_name="eric",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1", inputs={"value": "add_eric"}, placement_name="eric"
            )
        )

        compiler = Compiler(
            passes=[
                ReplicatedEncodingPass(),
                ReplicatedOpsPass(),
                ReplicatedLoweringPass(),
            ]
        )
        comp = compiler.run_passes(comp)

        assert all(
            isinstance(comp.placement(op.placement_name), HostPlacement)
            for op in comp.operations.values()
        )

    def test_replicated_mul_lowering(self):

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
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input",
                inputs={},
                value=2,
                placement_name="bob",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.MulOperation(
                name="secure_mul",
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                placement_name="rep",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "secure_mul", "rhs": "secure_mul"},
                placement_name="dave",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "add_dave"}, placement_name="dave"
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "secure_mul", "rhs": "secure_mul"},
                placement_name="eric",
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1", inputs={"value": "add_eric"}, placement_name="eric"
            )
        )

        compiler = Compiler(
            passes=[
                ReplicatedEncodingPass(),
                ReplicatedOpsPass(),
                ReplicatedLoweringPass(),
            ]
        )
        comp = compiler.run_passes(comp)

        assert all(
            isinstance(comp.placement(op.placement_name), HostPlacement)
            for op in comp.operations.values()
        )


if __name__ == "__main__":
    unittest.main()
