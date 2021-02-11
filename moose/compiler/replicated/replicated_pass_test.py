import unittest

from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.compiler.replicated.encoding_pass import ReplicatedEncodingPass
from moose.compiler.replicated.replicated_pass import ReplicatedOpsPass
from moose.computation import fixedpoint as fixed_dialect
from moose.computation import replicated as rep_dialect
from moose.computation import standard as std_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.edsl import dtypes


class ReplicatedTest(parameterized.TestCase):
    def test_share_reveal_pass(self):

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
            std_dialect.ConstantOperation(
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            std_dialect.ConstantOperation(
                name="bob_input",
                inputs={},
                value=2,
                placement_name="bob",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            std_dialect.AddOperation(
                name="add",
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                placement_name="rep",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            std_dialect.AddOperation(
                name="add_dave",
                inputs={"lhs": "add", "rhs": "add"},
                placement_name="dave",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            std_dialect.OutputOperation(
                name="output_0", inputs={"value": "add_dave"}, placement_name="dave"
            )
        )
        comp.add_operation(
            std_dialect.AddOperation(
                name="add_eric",
                inputs={"lhs": "add", "rhs": "add"},
                placement_name="eric",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            std_dialect.OutputOperation(
                name="output_1", inputs={"value": "add_eric"}, placement_name="eric"
            )
        )

        compiler = Compiler(passes=[ReplicatedEncodingPass(), ReplicatedOpsPass()])
        comp = compiler.run_passes(comp)

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
            std_dialect.ConstantOperation(
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            std_dialect.ConstantOperation(
                name="bob_input",
                inputs={},
                value=2,
                placement_name="bob",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            rep_dialect.SetupOperation(
                name="replicated_setup_0",
                inputs={},
                placement_name="rep",
                output_type=rep_dialect.ReplicatedSetupType(),
            )
        )
        expected_comp.add_operation(
            fixed_dialect.EncodeOperation(
                name="encode_2",
                inputs={"value": "alice_input"},
                placement_name="rep",
                output_type=fixed_dialect.EncodedTensorType(
                    datatype=dtypes.fixed(15, 16), precision=16
                ),
                precision=16,
            )
        )
        expected_comp.add_operation(
            fixed_dialect.EncodeOperation(
                name="encode_3",
                inputs={"value": "bob_input"},
                placement_name="rep",
                output_type=fixed_dialect.EncodedTensorType(
                    datatype=dtypes.fixed(15, 16), precision=16
                ),
                precision=16,
            )
        )
        expected_comp.add_operation(
            rep_dialect.ShareOperation(
                name="share_0",
                inputs={"setup": "replicated_setup_0", "value": "encode_2"},
                placement_name="rep",
                output_type=rep_dialect.ReplicatedTensorType(
                    dtype=dtypes.fixed(15, 16)
                ),
            )
        )
        expected_comp.add_operation(
            rep_dialect.ShareOperation(
                name="share_1",
                inputs={"setup": "replicated_setup_0", "value": "encode_3"},
                placement_name="rep",
                output_type=rep_dialect.ReplicatedTensorType(
                    dtype=dtypes.fixed(15, 16)
                ),
            )
        )
        expected_comp.add_operation(
            rep_dialect.AddOperation(
                name="replicated_add_0",
                inputs={
                    "setup": "replicated_setup_0",
                    "lhs": "share_0",
                    "rhs": "share_1",
                },
                placement_name="rep",
                output_type=rep_dialect.ReplicatedTensorType(
                    dtype=dtypes.fixed(15, 16)
                ),
            )
        )
        expected_comp.add_operation(
            rep_dialect.RevealOperation(
                name="reveal_0",
                inputs={"setup": "replicated_setup_0", "value": "replicated_add_0"},
                recipient_name="dave",
                placement_name="rep",
                output_type=fixed_dialect.EncodedTensorType(
                    datatype=dtypes.fixed(15, 16), precision=16
                ),
            )
        )
        expected_comp.add_operation(
            fixed_dialect.DecodeOperation(
                name="decode_2",
                inputs={"value": "reveal_0"},
                placement_name="rep",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
                precision=16,
            )
        )
        expected_comp.add_operation(
            std_dialect.AddOperation(
                name="add_dave",
                inputs={"lhs": "decode_2", "rhs": "decode_2"},
                placement_name="dave",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            std_dialect.OutputOperation(
                name="output_0", inputs={"value": "add_dave"}, placement_name="dave"
            )
        )
        expected_comp.add_operation(
            rep_dialect.RevealOperation(
                name="reveal_1",
                inputs={"setup": "replicated_setup_0", "value": "replicated_add_0"},
                recipient_name="eric",
                placement_name="rep",
                output_type=fixed_dialect.EncodedTensorType(
                    datatype=dtypes.fixed(15, 16), precision=16
                ),
            )
        )
        expected_comp.add_operation(
            fixed_dialect.DecodeOperation(
                name="decode_3",
                inputs={"value": "reveal_1"},
                placement_name="rep",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
                precision=16,
            )
        )
        expected_comp.add_operation(
            std_dialect.AddOperation(
                name="add_eric",
                inputs={"lhs": "decode_3", "rhs": "decode_3"},
                placement_name="eric",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            std_dialect.OutputOperation(
                name="output_1", inputs={"value": "add_eric"}, placement_name="eric"
            )
        )

        assert comp.placements == expected_comp.placements
        assert comp.operations == expected_comp.operations
        assert comp == expected_comp


if __name__ == "__main__":
    unittest.main()
