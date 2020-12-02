import numpy as np
from moose import ring_add, ring_mul, ring_sub


def test_ring_add_usual():
    a = np.array([1, 2, 3], dtype=np.uint64)
    b = np.array([4, 5, 6], dtype=np.uint64)
    c1 = a + b
    c2 = ring_add(a, b)

    np.testing.assert_array_equal(c1, c2)


def test_ring_mul_usual():
    a = np.array([1, 2, 3], dtype=np.uint64)
    b = np.array([4, 5, 6], dtype=np.uint64)
    c1 = a * b
    c2 = ring_mul(a, b)

    np.testing.assert_array_equal(c1, c2)


def test_ring_sub_usual():
    a = np.array([1, 2, 3], dtype=np.uint64)
    b = np.array([4, 5, 6], dtype=np.uint64)
    c1 = b - a
    c2 = ring_sub(b, a)

    np.testing.assert_array_equal(c1, c2)
