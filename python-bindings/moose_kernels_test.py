import numpy as np
from moose_kernels import ring_add

def test_ring_add_usual():
  a = np.array([1,2,3], dtype=np.uint64)
  b = np.array([4,5,6], dtype=np.uint64)
  c1 = a + b
  c2 = ring_add(a, b)

  np.testing.assert_array_equal(c1, c2)
