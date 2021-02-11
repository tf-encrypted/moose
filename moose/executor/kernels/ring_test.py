import unittest

import numpy as np
from absl.testing import parameterized

from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.edsl import dtypes
from moose.testing import run_test_computation


class RingKernelTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            lambda plc: ring_dialect.RingAddOperation(
                name="ring_add",
                placement_name=plc.name,
                inputs={"lhs": "x", "rhs": "y"},
            ),
            "ring_add",
            lambda x, y: x + y,
        ),
        (
            lambda plc: ring_dialect.RingMulOperation(
                name="ring_mul",
                placement_name=plc.name,
                inputs={"lhs": "x", "rhs": "y"},
            ),
            "ring_mul",
            lambda x, y: x * y,
        ),
        (
            lambda plc: ring_dialect.RingSubOperation(
                name="ring_sub",
                placement_name=plc.name,
                inputs={"lhs": "x", "rhs": "y"},
            ),
            "ring_sub",
            lambda x, y: x - y,
        ),
    )
    def test_ring_binop_invocation(self, ring_op_lmbd, ring_op_name, numpy_lmbd):
        a = np.array([3], dtype=np.uint64)
        b = np.array([2], dtype=np.uint64)
        c = numpy_lmbd(a, b)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                value=a,
                output_type=standard_dialect.TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="y",
                placement_name=alice.name,
                inputs={},
                value=b,
                output_type=standard_dialect.TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(ring_op_lmbd(alice))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": ring_op_name},
            )
        )

        results = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(c, results[alice]["z"])

    @parameterized.parameters(
        ([[1, 2], [3, 4]], [[1, 0], [0, 1]]),
        ([[1, 2], [3, 4]], [1, 1]),
        ([1, 1], [[1, 2], [3, 4]]),
    )
    def test_ring_dot_invocation(self, a, b):
        x = np.array(a, dtype=np.uint64)
        y = np.array(b, dtype=np.uint64)
        exp = np.dot(x, y)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                value=x,
                output_type=standard_dialect.TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="y",
                placement_name=alice.name,
                inputs={},
                value=y,
                output_type=standard_dialect.TensorType(dtype=dtypes.float32),
            )
        )
        comp.add_operation(
            ring_dialect.RingDotOperation(
                name="ring_dot",
                placement_name=alice.name,
                inputs={"lhs": "x", "rhs": "y"},
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "ring_dot"},
            )
        )

        results = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(exp, results[alice]["z"])

    def test_fill(self):
        expected = np.full((2, 2), 1)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x_shape",
                placement_name=alice.name,
                inputs={},
                value=(2, 2),
                output_type=standard_dialect.ShapeType(),
            )
        )
        comp.add_operation(
            ring_dialect.FillTensorOperation(
                name="x",
                placement_name=alice.name,
                value=1,
                inputs={"shape": "x_shape"},
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="x_filled",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "x"},
            )
        )

        results = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(expected, results[alice]["x_filled"])

    def test_sum(self):
        x = np.array([[1, 2], [3, 4]], dtype=np.uint64)
        expected = np.sum(x, axis=0)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                value=x,
                output_type=standard_dialect.TensorType(dtype=dtypes.uint64),
            )
        )
        comp.add_operation(
            ring_dialect.RingSumOperation(
                name="sum", placement_name=alice.name, axis=0, inputs={"x": "x"},
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="z",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "sum"},
            )
        )

        results = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(expected, results[alice]["z"])

    def test_bitwise_ops(self):
        expected = np.array([2, 2], dtype=np.uint64)

        x = np.array([4, 4], dtype=np.uint64)

        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x",
                placement_name=alice.name,
                inputs={},
                value=x,
                output_type=standard_dialect.TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            ring_dialect.RingShrOperation(
                name="ring_shr",
                placement_name=alice.name,
                inputs={"value": "x"},
                amount=1,
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=alice.name,
                value="x_shifted",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "ring_shr"},
            )
        )

        results = run_test_computation(comp, [alice])
        np.testing.assert_array_equal(expected, results[alice]["x_shifted"])


if __name__ == "__main__":
    unittest.main()
