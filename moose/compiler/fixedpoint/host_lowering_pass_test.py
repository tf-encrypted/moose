import unittest

from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.compiler.fixedpoint import host_lowering_pass
from moose.computation import dtypes
from moose.computation import fixedpoint as fixedpoint_ops
from moose.computation import standard as standard_ops
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import TensorType


class HostLoweringPassTest(parameterized.TestCase):
    def test_mul_lowering(self):
        comp = Computation(placements={}, operations={})

        comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_input",
                inputs={},
                value=2,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="x_encode",
                placement_name="alice",
                inputs={"value": "x_input"},
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 23), precision=23,
                ),
                precision=23,
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="y_input",
                inputs={},
                value=3,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="y_encode",
                placement_name="alice",
                inputs={"value": "y_input"},
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 23), precision=23,
                ),
                precision=23,
            )
        )
        comp.add_operation(
            standard_ops.MulOperation(
                name="x_y_prod",
                placement_name="alice",
                inputs={"lhs": "x_encode", "rhs": "y_encode"},
                output_type=TensorType(dtype=dtypes.fixed(14, 23)),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "x_y_prod"}, placement_name="alice",
            )
        )

        compiler = Compiler(passes=[host_lowering_pass.HostLoweringPass()])
        comp = compiler.run_passes(comp)

        expected_comp = Computation(placements={}, operations={})
        expected_comp.add_placement(HostPlacement(name="alice"))
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_input",
                inputs={},
                value=2,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="x_encode",
                placement_name="alice",
                inputs={"value": "x_input"},
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 23), precision=23,
                ),
                precision=23,
            )
        )
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="y_input",
                inputs={},
                value=3,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        expected_comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="y_encode",
                placement_name="alice",
                inputs={"value": "y_input"},
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 23), precision=23,
                ),
                precision=23,
            )
        )
        expected_comp.add_operation(
            fixedpoint_ops.MulOperation(
                name="fixed_mul_0",
                placement_name="alice",
                inputs={"lhs": "x_encode", "rhs": "y_encode"},
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 46), precision=46,
                ),
            )
        )
        expected_comp.add_operation(
            fixedpoint_ops.TruncOperation(
                name="fixed_trunc_0",
                placement_name="alice",
                inputs={"value": "fixed_mul_0"},
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 23), precision=23,
                ),
                precision=23,
            )
        )
        expected_comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "fixed_trunc_0"},
                placement_name="alice",
            )
        )
        assert comp.operations == expected_comp.operations
        assert comp == expected_comp


if __name__ == "__main__":
    unittest.main()
