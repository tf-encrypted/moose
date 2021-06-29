import random

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from pymoose import MooseLocalRuntime
from pymoose import moose_kernels as mkls

from moose import edsl
from moose.computation.standard import StringType
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


x_owner = edsl.host_placement(name="x_owner")
y_owner = edsl.host_placement(name="y_owner")
output_owner = edsl.host_placement("output_owner")


@edsl.computation
def add_full_storage(
    x_key: edsl.Argument(x_owner, vtype=StringType()),
    y_key: edsl.Argument(y_owner, vtype=StringType()),
):
    with x_owner:
        x = edsl.load(x_key, dtype=edsl.float64)
    with y_owner:
        y = edsl.load(y_key, dtype=edsl.float64)
    with output_owner:
        out = edsl.add(x, y)
        res = edsl.save("output", out)
    return res


@edsl.computation
def add_input_storage(
    x_key: edsl.Argument(x_owner, vtype=StringType()),
    y_key: edsl.Argument(y_owner, vtype=StringType()),
):
    with x_owner:
        x = edsl.load(x_key, dtype=edsl.float64)
    with y_owner:
        y = edsl.load(y_key, dtype=edsl.float64)
    with output_owner:
        out = edsl.add(x, y)
    return out


@edsl.computation
def add_output_storage(
    x: edsl.Argument(x_owner, dtype=edsl.float64),
    y: edsl.Argument(y_owner, dtype=edsl.float64),
):
    with output_owner:
        out = edsl.add(x, y)
        res = edsl.save(out, "output")
    return res


@edsl.computation
def add_no_storage(
    x: edsl.Argument(x_owner, dtype=edsl.float64),
    y: edsl.Argument(y_owner, dtype=edsl.float64),
):
    with output_owner:
        out = edsl.add(x, y)
    return out


@edsl.computation
def add_multioutput(
    x: edsl.Argument(x_owner, dtype=edsl.float64),
    y: edsl.Argument(y_owner, dtype=edsl.float64),
):
    with output_owner:
        out = edsl.add(x, y)
    return out, x, y


class RunComputation(parameterized.TestCase):
    def setUp(self):
        self.x_input = {"x": np.array([1.0], dtype=np.float64)}
        self.y_input = {"y": np.array([2.0], dtype=np.float64)}
        self.storage_dict = {
            "x_owner": self.x_input,
            "y_owner": self.y_input,
            "output_owner": {},
        }
        self.empty_storage = {
            "x_owner": {},
            "y_owner": {},
            "output_owner": {},
        }
        self.storage_args = {"x_key": "x", "y_key": y}
        self.actual_args = {**self.x_input, **self.y_input}

    def _inner_prepare_runtime(self, comp, storage_dict):
        concrete_comp = edsl.trace_and_compile(comp, ring=128)
        comp_bin = serialize_computation(concrete_comp)
        runtime = MooseLocalRuntime(executors_storage=storage_dict)
        return comp_bin, runtime

    def test_full_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_full_storage, self.storage_dict
        )
        outputs = runtime.evaluate_computation(comp_bin, self.storage_args)
        assert len(outputs) == 0
        result = runtime.get_value_from_storage("output_owner", "output")
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_input_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_input_storage, self.storage_dict
        )
        result = runtime.evaluate_computation(comp_bin, self.storage_args)
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_output_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_output_storage, self.empty_storage
        )
        outputs = runtime.evaluate_computation(comp_bin, self.actual_args)
        assert len(outputs) == 0
        result = runtime.get_value_from_storage("output_owner", "output")
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_no_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_no_storage, self.storage_dict
        )
        result = runtime.evaluate_computation(comp_bin, self.actual_args)
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_multioutput(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_no_storage, self.storage_dict
        )
        (out, x, y) = runtime.evaluate_computation(comp_bin, self.actual_args)
        np.testing.assert_array_equal(x, np.array([1.0]))
        np.testing.assert_array_equal(y, np.array([2.0]))
        np.testing.assert_array_equal(out, np.array([3.0]))


if __name__ == "__main__":
    absltest.main()
