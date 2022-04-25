import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.computation import types as ty
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class SoftmaxExample(parameterized.TestCase):
    def _setup_comp(self, axis, axis_idx_max, replicated=True):
        bob = edsl.host_placement(name="bob")

        @edsl.computation
        def my_comp(
            x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
        ):
            with bob:
                x = edsl.load(x_uri, dtype=edsl.float64)
                x_soft = edsl.softmax(x, axis=axis, upmost_index=axis_idx_max)
                res = edsl.save("softmax", x_soft)
            return res

        return my_comp

    @parameterized.parameters(
        ([10, 11, 12], 0, 2),
        ([3, 4, 3], 0, 2),
        ([-0.64, 0.76, 0.97], 0, 2),
        ([0.02, -0.04, 1.08, 0.01], 0, 3),
        ([-1.35, -1.34, -0.72, -0.21], 0, 3),
        ([-0.61964023, -0.9119955, 1.50079676, -1.46759315], 0, 3),
        ([-0.08150293, -1.50330937, -0.99238243, -2.65759917], 0, 3),
    )
    def test_example_execute(self, x, axis, axis_idx_max):
        comp_host = self._setup_comp(axis, axis_idx_max)
        traced_softmax_comp = edsl.trace(comp_host)
        x_arg = np.array(x, dtype=np.float64)
        storage = {
            "bob": {"x_arg": x_arg},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_softmax_comp,
            role_assignment={"bob": "bob"},
            arguments={"x_uri": "x_arg"},
        )
        softmax_runtime_host = runtime.read_value_from_storage("bob", "softmax")

        ex = np.exp(x_arg - x_arg.max(axis=axis, keepdims=True))
        softmax_numpy = ex / np.sum(ex, axis=axis, keepdims=True)

        np.testing.assert_almost_equal(softmax_runtime_host, softmax_numpy, decimal=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="softmax example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
