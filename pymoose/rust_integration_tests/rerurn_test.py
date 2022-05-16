import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class RerunExample(parameterized.TestCase):
    """A test to showcase re-executing the computation.

    This test showcases the ability to execute more than one computation
    on the same instance of a LocalMooseRuntime and the case in which a
    new instance is needed.
    """

    def _setup_comp(self):
        """Just a demo computation"""

        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        mir3 = edsl.mirrored_placement(name="mir3", players=[alice, bob, carole])

        @edsl.computation
        def my_comp():
            with mir3:
                x = edsl.constant(np.array([1.5, 2.3, 3, 3], dtype=np.float64))
                x = edsl.cast(x, dtype=edsl.fixed(8, 27))
            with alice:
                y = edsl.cast(x, dtype=edsl.float64)

            return y

        return my_comp

    def test_example_execute(self):
        comp = self._setup_comp()
        traced_less_comp = edsl.trace(comp)
        runtime = LocalMooseRuntime(identities=["alice", "bob", "carole"])
        result_dict = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )

        actual_result = list(result_dict.values())[0]

        np.testing.assert_almost_equal(actual_result[0], 1.5)
        np.testing.assert_almost_equal(actual_result[1], 2.3)
        np.testing.assert_almost_equal(actual_result[2], 3)

        # You should be able to rerun the computaiton as-is. You'll get a fresh session.
        result_dict = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )

        # You can also remap the roles among the original identities
        result_dict = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={"alice": "bob", "bob": "carole", "carole": "alice"},
            arguments={},
        )

        # But if you want to have different identities, you would need a new instance
        runtime = LocalMooseRuntime(identities=["newalice", "newbob", "newcarole"])
        result_dict = runtime.evaluate_computation(
            computation=traced_less_comp,
            role_assignment={
                "alice": "newalice",
                "bob": "newbob",
                "carole": "newcarole",
            },
            arguments={},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
