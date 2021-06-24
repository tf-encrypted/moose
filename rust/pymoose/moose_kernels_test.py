import random

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from pymoose import MooseRuntime
from pymoose import moose_kernels as mkls

from moose import edsl
from moose.computation.utils import serialize_computation


class BinaryOp(parameterized.TestCase):
    @parameterized.parameters(
        (lambda x, y: x + y, mkls.ring_add),
        (lambda x, y: x * y, mkls.ring_mul),
        (lambda x, y: x - y, mkls.ring_sub),
    )
    def test_usual_binary_op(self, numpy_lmbd, moose_op):
        a = np.array([1, 2, 3], dtype=np.uint64)
        b = np.array([4, 5, 6], dtype=np.uint64)

        c1 = numpy_lmbd(b, a)
        c2 = moose_op(b, a)

        np.testing.assert_array_equal(c1, c2)

    def test_shape(self):
        a = np.array([1, 2, 3], dtype=np.uint64)
        assert mkls.ring_shape(a) == [3]

    @parameterized.parameters(
        ([[1, 2], [3, 4]], [[1, 0], [0, 1]]),
        ([[1, 2], [3, 4]], [1, 1]),
        ([1, 1], [[1, 2], [3, 4]]),
    )
    def test_dot_prod(self, a, b):
        x = np.array(a, dtype=np.uint64)
        y = np.array(b, dtype=np.uint64)
        exp = np.dot(x, y)
        res = mkls.ring_dot(x, y)
        np.testing.assert_array_equal(res, exp)


class SumOp(parameterized.TestCase):
    @parameterized.parameters([0, 1, None])
    def test_sum_op(self, axis):
        x = np.array([[1, 2], [3, 4]], dtype=np.uint64)
        actual = mkls.ring_sum(x, axis=axis)
        expected = np.sum(x, axis=axis)
        np.testing.assert_array_equal(actual, expected)


class SamplingOperations(parameterized.TestCase):
    def test_sample_key(self):
        key = mkls.sample_key()
        assert len(key) == 16
        assert isinstance(key, bytes)

    @parameterized.parameters((b"0"), (b"1"), (b"123456"))
    def test_expand_seed(self, nonce):
        key = mkls.sample_key()
        seed0 = mkls.derive_seed(key, nonce)
        seed1 = mkls.derive_seed(key, nonce)

        assert len(seed0) == 16
        assert len(seed1) == 16

        assert isinstance(seed0, bytes)
        assert isinstance(seed1, bytes)

        # check determinism
        assert seed0 == seed1

        # check non-determinism
        assert mkls.derive_seed(mkls.sample_key(), nonce) != mkls.derive_seed(
            mkls.sample_key(), nonce
        )
        assert mkls.derive_seed(
            key, random.randint(0, 2 ** 128).to_bytes(16, byteorder="little")
        ) != mkls.derive_seed(
            key, random.randint(0, 2 ** 128).to_bytes(16, byteorder="little")
        )

    def test_sample(self):
        actual = mkls.ring_sample((2, 2), mkls.sample_key())
        assert mkls.ring_shape(actual) == [2, 2]
        random_bits = mkls.ring_sample((2, 2), mkls.sample_key(), max_value=1)
        assert np.all(random_bits <= 1)
        assert np.all(0 <= random_bits)


class FillOp(parameterized.TestCase):
    def test_fill_op(self):
        actual = mkls.ring_fill((2, 2), 1)
        expected = np.full((2, 2), 1, dtype=np.uint64)
        np.testing.assert_array_equal(actual, expected)


class RingBitOps(parameterized.TestCase):
    def test_bitwise_ops(self):
        a = np.array([2 ** i for i in range(64)], dtype=np.uint64)
        for i in range(10):
            np.testing.assert_array_equal(a << i, mkls.ring_shl(a, i))
            np.testing.assert_array_equal(a >> i, mkls.ring_shr(a, i))


class BitTensorOps(parameterized.TestCase):
    @parameterized.parameters(
        ([[0, 1], [0, 1]], [[1, 0], [1, 0]]),
        ([0], [0]),
        ([0], [1]),
        ([1], [0]),
        ([1], [1]),
    )
    def test_bitwise_ops(self, a, b):
        x = np.array(a, dtype=np.uint8)
        y = np.array(b, dtype=np.uint8)

        np.testing.assert_array_equal(x & y, mkls.bit_and(x, y))
        np.testing.assert_array_equal(x ^ y, mkls.bit_xor(x, y))

    @parameterized.parameters(([0]), ([1]), [[0, 1, 1]])
    def test_ring_inject(self, a):
        x = np.array(a, dtype=np.uint8)
        np.testing.assert_array_equal(x, mkls.ring_inject(x, 0))
        np.testing.assert_array_equal(x << 1, mkls.ring_inject(x, 1))
        np.testing.assert_array_equal(x << 2, mkls.ring_inject(x, 2))

    def test_shape(self):
        a = np.array([1, 2, 3], dtype=np.uint8)
        assert mkls.bit_shape(a) == [3]


class RunComputation(parameterized.TestCase):
    def test_run_computation(self):
        def _build_computation():
            x_owner = edsl.host_placement(name="x_owner")
            y_owner = edsl.host_placement(name="y_owner")
            output_owner = edsl.host_placement("output_owner")

            @edsl.computation
            def add_comp():

                with x_owner:
                    x = edsl.load("x_data", dtype=edsl.float32)

                with y_owner:
                    y = edsl.load("y_data", dtype=edsl.float32)

                with output_owner:
                    out = edsl.add(x, y)
                    res = edsl.save("output", out)

                return res

            concrete_comp = edsl.trace_and_compile(add_comp, ring=128)
            return concrete_comp

        comp = _build_computation()
        comp_bin = serialize_computation(comp)
        storages = {
            "x_owner": {"x_data": np.array([1.0])},
            "y_owner": {"y_data": np.array([2.0])},
            "output_owner": {},
        }
        args = {"": ""}

        runtime = MooseRuntime(storages)
        runtime.evaluate_computation(comp_bin, args)
        result = runtime.get_value_from_storage("output_owner", "output")
        np.testing.assert_array_equal(result, np.array([3.0]))


if __name__ == "__main__":
    absltest.main()
