import unittest

from absl.testing import parameterized

from moose.compiler.compiler import Compiler
from moose.compiler.fixedpoint import ring_lowering_pass
from moose.computation import dtypes
from moose.computation import fixedpoint as fixedpoint_ops
from moose.computation import ring as ring_ops
from moose.computation import standard as standard_ops
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.standard import TensorType


class RingLoweringPassTest(parameterized.TestCase):
    def test_fixed_encode_lowering(self):
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
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "x_encode"}, placement_name="alice",
            )
        )

        compiler = Compiler(passes=[ring_lowering_pass.RingLoweringPass()])
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
            fixedpoint_ops.RingEncodeOperation(
                name="ring_encode_0",
                placement_name="alice",
                inputs={"value": "x_input"},
                scaling_factor=2 ** 23,
            )
        )
        expected_comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "ring_encode_0"},
                placement_name="alice",
            )
        )
        assert comp.operations == expected_comp.operations
        assert comp == expected_comp

    def test_fixed_decode_lowering(self):
        comp = Computation(placements={}, operations={})

        comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_encoded",
                inputs={},
                value=13421772800,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(8, 27)),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="x_decode",
                placement_name="alice",
                inputs={"value": "x_encoded"},
                output_type=TensorType(dtypes.float64),
                precision=27,
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "x_decode"}, placement_name="alice",
            )
        )

        compiler = Compiler(passes=[ring_lowering_pass.RingLoweringPass()])
        comp = compiler.run_passes(comp)

        expected_comp = Computation(placements={}, operations={})
        expected_comp.add_placement(HostPlacement(name="alice"))
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_encoded",
                inputs={},
                value=13421772800,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(8, 27)),
            )
        )
        expected_comp.add_operation(
            fixedpoint_ops.RingDecodeOperation(
                name="ring_decode_0",
                placement_name="alice",
                inputs={"value": "x_encoded"},
                output_type=dtypes.float64,
                scaling_factor=2 ** 27,
            )
        )
        expected_comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "ring_decode_0"},
                placement_name="alice",
            )
        )

    def test_fixed_truncated_sum_lowering(self):
        comp = Computation(placements={}, operations={})

        comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_input",
                inputs={},
                value=2,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(14, 23)),
            )
        )
        comp.add_operation(
            fixedpoint_ops.SumOperation(
                name="fixed_sum_0",
                placement_name="alice",
                inputs={"x": "x_input"},
                axis=0,
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 23), precision=23,
                ),
            )
        )
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "fixed_sum_0"},
                placement_name="alice",
            )
        )

        compiler = Compiler(passes=[ring_lowering_pass.RingLoweringPass()])
        comp = compiler.run_passes(comp)

        expected_comp = Computation(placements={}, operations={})
        expected_comp.add_placement(HostPlacement(name="alice"))
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_input",
                inputs={},
                value=2,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(14, 23)),
            )
        )
        expected_comp.add_operation(
            ring_ops.RingSumOperation(
                name="ring_sum_0",
                placement_name="alice",
                inputs={"x": "x_input"},
                axis=0,
            )
        )
        expected_comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "ring_sum_0"}, placement_name="alice",
            )
        )
        assert comp.operations == expected_comp.operations
        assert comp == expected_comp

    def test_fixed_truncated_mul_lowering(self):
        comp = Computation(placements={}, operations={})

        comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_input",
                inputs={},
                value=2,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(14, 23)),
            )
        )
        comp.add_operation(
            standard_ops.ConstantOperation(
                name="y_input",
                inputs={},
                value=3,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(14, 23)),
            )
        )
        comp.add_operation(
            fixedpoint_ops.MulOperation(
                name="fixed_mul_0",
                placement_name="alice",
                inputs={"lhs": "x_input", "rhs": "y_input"},
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=dtypes.fixed(14, 46), precision=46,
                ),
            )
        )
        comp.add_operation(
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
        comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0",
                inputs={"value": "fixed_trunc_0"},
                placement_name="alice",
            )
        )

        compiler = Compiler(passes=[ring_lowering_pass.RingLoweringPass()])
        comp = compiler.run_passes(comp)

        expected_comp = Computation(placements={}, operations={})
        expected_comp.add_placement(HostPlacement(name="alice"))
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="x_input",
                inputs={},
                value=2,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(14, 23)),
            )
        )
        expected_comp.add_operation(
            standard_ops.ConstantOperation(
                name="y_input",
                inputs={},
                value=3,
                placement_name="alice",
                output_type=TensorType(dtype=dtypes.fixed(14, 23)),
            )
        )
        expected_comp.add_operation(
            ring_ops.RingMulOperation(
                name="ring_mul_0",
                placement_name="alice",
                inputs={"lhs": "x_input", "rhs": "y_input"},
            )
        )
        expected_comp.add_operation(
            ring_ops.RingShrOperation(
                name="ring_shr_0",
                placement_name="alice",
                inputs={"value": "ring_mul_0"},
                amount=23,
            )
        )
        expected_comp.add_operation(
            standard_ops.OutputOperation(
                name="output_0", inputs={"value": "ring_shr_0"}, placement_name="alice",
            )
        )
        assert comp.operations == expected_comp.operations
        assert comp == expected_comp


if __name__ == "__main__":
    unittest.main()
