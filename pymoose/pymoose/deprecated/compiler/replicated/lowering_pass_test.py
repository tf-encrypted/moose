import unittest

from absl.testing import parameterized

from pymoose.computation import dtypes
from pymoose.computation import fixedpoint as fixedpoint_ops
from pymoose.computation import standard as standard_ops
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import UnitType
from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.deprecated.compiler.fixedpoint.host_ring_lowering_pass import (
    HostRingLoweringPass,
)
from pymoose.deprecated.compiler.replicated.encoding_pass import ReplicatedEncodingPass
from pymoose.deprecated.compiler.replicated.lowering_pass import ReplicatedLoweringPass
from pymoose.deprecated.compiler.replicated.replicated_pass import ReplicatedOpsPass


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
        fp_dtype = dtypes.fixed(8, 27)

        comp.add_operation(
            standard_ops.ConstantOperation(
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="alice_encode",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input",
                inputs={},
                value=2,
                placement_name="bob",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="bob_encode",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add",
                inputs={"lhs": "alice_encode", "rhs": "bob_encode"},
                placement_name="rep",
                output_type=TensorType(dtype=fp_dtype),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="dave_add_decode",
                inputs={"value": "add"},
                placement_name="dave",
                output_type=TensorType(dtype=dtypes.float32),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "dave_add_decode", "rhs": "dave_add_decode"},
                placement_name="dave",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "add_dave"},
                placement_name="dave",
                output_type=UnitType(),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="eric_add_decode",
                inputs={"value": "add"},
                placement_name="eric",
                output_type=TensorType(dtype=dtypes.float32),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "eric_add_decode", "rhs": "eric_add_decode"},
                placement_name="eric",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1",
                inputs={"value": "add_eric"},
                placement_name="eric",
                output_type=UnitType(),
            )
        )

        compiler = Compiler(
            passes=[
                ReplicatedEncodingPass(),
                ReplicatedOpsPass(),
                HostRingLoweringPass(),
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
        fp_dtype = dtypes.fixed(8, 27)

        comp.add_operation(
            standard_ops.ConstantOperation(
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input",
                inputs={},
                value=2,
                placement_name="bob",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="alice_encode",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="bob_encode",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.MulOperation(
                name="secure_mul",
                inputs={"lhs": "alice_encode", "rhs": "bob_encode"},
                placement_name="rep",
                output_type=TensorType(dtype=fp_dtype),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="dave_decode",
                inputs={"value": "secure_mul"},
                placement_name="dave",
                output_type=TensorType(dtype=dtypes.float32),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "dave_decode", "rhs": "dave_decode"},
                placement_name="dave",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "add_dave"},
                placement_name="dave",
                output_type=UnitType(),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="eric_decode",
                inputs={"value": "secure_mul"},
                placement_name="eric",
                output_type=TensorType(dtype=dtypes.float32),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "eric_decode", "rhs": "eric_decode"},
                placement_name="eric",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1",
                inputs={"value": "add_eric"},
                placement_name="eric",
                output_type=UnitType(),
            )
        )

        compiler = Compiler(
            passes=[
                ReplicatedEncodingPass(),
                ReplicatedOpsPass(),
                HostRingLoweringPass(),
                ReplicatedLoweringPass(),
            ]
        )
        comp = compiler.run_passes(comp)

        assert all(
            isinstance(comp.placement(op.placement_name), HostPlacement)
            for op in comp.operations.values()
        )

    def test_replicated_dot_lowering(self):
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
            standard_ops.ConstantOperation(
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="bob_input",
                inputs={},
                value=2,
                placement_name="bob",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="alice_encode",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="bob_encode",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.DotOperation(
                name="secure_dot",
                inputs={"lhs": "alice_encode", "rhs": "bob_encode"},
                placement_name="rep",
                output_type=TensorType(dtype=fp_dtype),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="dave_decode",
                inputs={"value": "secure_dot"},
                placement_name="dave",
                output_type=TensorType(dtype=dtypes.float32),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_dave",
                inputs={"lhs": "dave_decode", "rhs": "dave_decode"},
                placement_name="dave",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "add_dave"},
                placement_name="dave",
                output_type=UnitType(),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="eric_decode",
                inputs={"value": "secure_dot"},
                placement_name="eric",
                output_type=TensorType(dtype=dtypes.float32),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.AddOperation(
                name="add_eric",
                inputs={"lhs": "eric_decode", "rhs": "eric_decode"},
                placement_name="eric",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_1",
                inputs={"value": "add_eric"},
                placement_name="eric",
                output_type=UnitType(),
            )
        )

        compiler = Compiler(
            passes=[
                ReplicatedEncodingPass(),
                ReplicatedOpsPass(),
                HostRingLoweringPass(),
                ReplicatedLoweringPass(),
            ]
        )
        comp = compiler.run_passes(comp)

        assert all(
            isinstance(comp.placement(op.placement_name), HostPlacement)
            for op in comp.operations.values()
        )

    def test_replicated_mean_lowering(self):

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
            standard_ops.ConstantOperation(
                name="alice_input",
                inputs={},
                value=1,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="alice_encode",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.MeanOperation(
                name="secure_mean",
                inputs={"x": "alice_encode"},
                axis=None,
                placement_name="rep",
                output_type=TensorType(dtype=fp_dtype),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="dave_decode",
                inputs={"value": "secure_mean"},
                placement_name="dave",
                output_type=TensorType(dtype=dtypes.float32),
                precision=fp_dtype.fractional_precision,
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "dave_decode"},
                placement_name="dave",
                output_type=UnitType(),
            )
        )

        compiler = Compiler(
            passes=[
                ReplicatedEncodingPass(),
                ReplicatedOpsPass(),
                HostRingLoweringPass(),
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
