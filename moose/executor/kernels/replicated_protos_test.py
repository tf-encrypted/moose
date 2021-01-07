import unittest

import numpy as np
from absl.testing import parameterized
from hypothesis import given
from hypothesis import strategies as st

from moose.compiler.compiler import Compiler
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import TensorType
from moose.runtime import TestRuntime as LocalRuntime

# to get 2 random lists of equal size using hypothesis
# https://stackoverflow.com/questions/51597021/python-hypothesis-ensure-that-input-lists-have-same-length

pair_lists = st.lists(
    st.tuples(
        st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100)
    ),
    min_size=1,
)


class ReplicatedProtocolsTest(parameterized.TestCase):
    @parameterized.parameters(
        (lambda x, y: x + y, standard_dialect.AddOperation),
        (lambda x, y: x - y, standard_dialect.SubOperation),
        # the following will work only after we can do fix point multiplication
        # without special encoding
        # (lambda x, y: x * y, standard_dialect.MulOperation),
    )
    @given(pair_lists)
    def test_bin_op(self, numpy_lmbd, replicated_std_op, bin_args):
        comp = Computation(operations={}, placements={})

        alice = HostPlacement(name="alice")
        bob = HostPlacement(name="bob")
        carole = HostPlacement(name="carole")
        rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])

        comp.add_placement(alice)
        comp.add_placement(bob)
        comp.add_placement(carole)
        comp.add_placement(rep)

        a, b = map(list, zip(*bin_args))
        x = np.array(a, dtype=np.float64)
        y = np.array(b, dtype=np.float64)

        z = numpy_lmbd(x, y)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=x,
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="bob_input",
                value=y,
                placement_name=bob.name,
                inputs={},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            replicated_std_op(
                name="rep_op",
                placement_name=rep.name,
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"value": "rep_op"},
                placement_name=carole.name,
                key="result",
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output", placement_name=carole.name, inputs={"value": "save"},
            )
        )

        compiler = Compiler()

        comp = compiler.run_passes(comp)

        placement_instantiation = {
            alice: alice.name,
            bob: bob.name,
            carole: carole.name,
        }

        runtime = LocalRuntime()
        runtime.evaluate_computation(
            computation=comp, placement_instantiation=placement_instantiation
        )

        np.testing.assert_array_equal(
            z, runtime.get_executor(carole.name).store["result"]
        )


if __name__ == "__main__":
    unittest.main()
