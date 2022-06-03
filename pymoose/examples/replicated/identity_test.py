import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger


class TensorIdentityExample(parameterized.TestCase):
    def _setup_placements(self):
        a0 = pm.host_placement(name="alice-0")
        a1 = pm.host_placement(name="alice-0")
        b0 = pm.host_placement(name="bob-0")
        b1 = pm.host_placement(name="bob-1")
        c0 = pm.host_placement(name="carole-0")
        c1 = pm.host_placement(name="carole-1")
        r0 = pm.replicated_placement("replicated-0", [a0, b0, c0])
        r1 = pm.replicated_placement("replicated-1", [a1, b1, c1])
        return {
            "alice-0": a0,
            "bob-0": b0,
            "carole-0": c0,
            "alice-1": a1,
            "bob-1": b1,
            "carole-1": c1,
            "replicated-0": r0,
            "replicated-1": r1,
        }

    def _setup_identity_comp(self, from_, to_, encoded_tensor):
        placement_dict = self._setup_placements()
        from_plc = placement_dict[from_]
        to_plc = placement_dict[to_]
        input_plc = placement_dict["bob-0"]
        output_plc = placement_dict["carole-0"]

        @pm.computation
        def identity_comp():
            with input_plc:
                x = pm.constant(np.array([2], dtype=np.float64))
                zero = pm.constant(np.array([0], dtype=np.float64))
                if encoded_tensor:
                    x = pm.cast(x, dtype=pm.fixed(8, 27))
                    zero = pm.cast(zero, dtype=pm.fixed(8, 27))

            with from_plc:
                x = pm.identity(x)

            with to_plc:
                x = pm.identity(x)

            with output_plc:
                x = pm.add(x, zero)  # "send" tensor without using pm.identity codepath
                if encoded_tensor:
                    x = pm.cast(x, dtype=pm.float64)

            return x

        return identity_comp

    @parameterized.parameters(
        ("alice-0", "alice-1", True),
        ("alice-0", "alice-1", False),
        ("alice-0", "replicated-0", True),
        ("replicated-0", "replicated-1", True),
        ("replicated-0", "alice-0", True),
    )
    def test_identity_example_execute(self, f, t, e):
        identity_comp = self._setup_identity_comp(f, t, e)
        identities = [
            "alice-0",
            "bob-0",
            "carole-0",
            "alice-1",
            "bob-1",
            "carole-1",
        ]
        runtime = pm.LocalMooseRuntime(identities)
        result_dict = runtime.evaluate_computation(
            computation=identity_comp,
            arguments={},
        )
        actual_result = list(result_dict.values())[0]
        np.testing.assert_almost_equal(actual_result, np.array([2], dtype=np.float64))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
