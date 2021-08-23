import unittest

from absl.testing import parameterized

from pymoose.computation import dtypes
from pymoose.computation import replicated as rep_dialect
from pymoose.computation import standard as std_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import UnitType
from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.deprecated.compiler.replicated.encoding_pass import ReplicatedEncodingPass
from pymoose.deprecated.compiler.replicated.replicated_pass import ReplicatedOpsPass
from pymoose.deprecated.computation import fixedpoint as fixed_dialect
from pymoose.deprecated.computation import fixedpoint as fixedpoint_ops


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

        fp_dtype = dtypes.fixed(8, 27)

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
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
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
            fixedpoint_ops.EncodeOperation(
                name="encode_bob",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            std_dialect.AddOperation(
                name="add",
                inputs={"lhs": "encode_alice", "rhs": "encode_bob"},
                placement_name="rep",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_dave",
                inputs={"value": "add"},
                placement_name="dave",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            std_dialect.AddOperation(
                name="add_dave",
                inputs={"lhs": "decode_dave", "rhs": "decode_dave"},
                placement_name="dave",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            std_dialect.OutputOperation(
                name="output_0",
                inputs={"value": "add_dave"},
                placement_name="dave",
                output_type=UnitType(),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_eric",
                inputs={"value": "add"},
                placement_name="eric",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            std_dialect.AddOperation(
                name="add_eric",
                inputs={"lhs": "decode_eric", "rhs": "decode_eric"},
                placement_name="eric",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            std_dialect.OutputOperation(
                name="output_1",
                inputs={"value": "add_eric"},
                placement_name="eric",
                output_type=UnitType(),
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
        expected_encoded_dtype = dtypes.fixed(8, 27)

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
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
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
            fixedpoint_ops.EncodeOperation(
                name="encode_bob",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
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
            rep_dialect.ShareOperation(
                name="share_0",
                inputs={"setup": "replicated_setup_0", "value": "encode_alice"},
                placement_name="rep",
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=expected_encoded_dtype
                ),
            )
        )
        expected_comp.add_operation(
            rep_dialect.ShareOperation(
                name="share_1",
                inputs={"setup": "replicated_setup_0", "value": "encode_bob"},
                placement_name="rep",
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=expected_encoded_dtype
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
                output_type=rep_dialect.ReplicatedRingTensorType(
                    dtype=expected_encoded_dtype,
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
                    dtype=expected_encoded_dtype,
                    precision=expected_encoded_dtype.fractional_precision,
                ),
            )
        )
        expected_comp.add_operation(
            fixed_dialect.DecodeOperation(
                name="decode_dave",
                inputs={"value": "reveal_0"},
                placement_name="dave",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
                precision=expected_encoded_dtype.fractional_precision,
            )
        )
        expected_comp.add_operation(
            std_dialect.AddOperation(
                name="add_dave",
                inputs={"lhs": "decode_dave", "rhs": "decode_dave"},
                placement_name="dave",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            std_dialect.OutputOperation(
                name="output_0",
                inputs={"value": "add_dave"},
                placement_name="dave",
                output_type=UnitType(),
            )
        )
        expected_comp.add_operation(
            rep_dialect.RevealOperation(
                name="reveal_1",
                inputs={"setup": "replicated_setup_0", "value": "replicated_add_0"},
                recipient_name="eric",
                placement_name="rep",
                output_type=fixed_dialect.EncodedTensorType(
                    dtype=expected_encoded_dtype,
                    precision=expected_encoded_dtype.fractional_precision,
                ),
            )
        )
        expected_comp.add_operation(
            fixed_dialect.DecodeOperation(
                name="decode_eric",
                inputs={"value": "reveal_1"},
                placement_name="eric",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
                precision=expected_encoded_dtype.fractional_precision,
            )
        )
        expected_comp.add_operation(
            std_dialect.AddOperation(
                name="add_eric",
                inputs={"lhs": "decode_eric", "rhs": "decode_eric"},
                placement_name="eric",
                output_type=std_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            std_dialect.OutputOperation(
                name="output_1",
                inputs={"value": "add_eric"},
                placement_name="eric",
                output_type=UnitType(),
            )
        )

        assert comp.placements == expected_comp.placements
        assert comp.operations == expected_comp.operations
        assert comp == expected_comp


if __name__ == "__main__":
    unittest.main()
