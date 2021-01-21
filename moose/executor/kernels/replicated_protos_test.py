import itertools
import unittest
from datetime import timedelta

import numpy as np
from absl.testing import parameterized
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from moose.compiler.compiler import Compiler
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import TensorType
from moose.testing import run_test_computation

# to get 2 random lists of equal size using hypothesis
# https://stackoverflow.com/questions/51597021/python-hypothesis-ensure-that-input-lists-have-same-length

pair_lists = st.lists(
    st.tuples(
        st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100)
    ),
    min_size=1,
)

dim = st.integers(min_value=2, max_value=12)
a_dims = st.tuples(st.shared(dim), st.shared(dim, key="inner_dim"))
b_dims = st.tuples(st.shared(dim, key="inner_dim"), st.shared(dim))
dotprod_inputs = st.tuples(
    hnp.arrays(
        dtype=np.float64,
        shape=a_dims,
        elements=st.floats(
            min_value=1, max_value=5, allow_infinity=False, allow_nan=False, width=16
        ),
    ),
    hnp.arrays(
        dtype=np.float64,
        shape=b_dims,
        elements=st.floats(
            min_value=1, max_value=5, allow_infinity=False, allow_nan=False, width=16
        ),
    ),
)


def _setup_replicated_computation(comp):
    alice = HostPlacement(name="alice")
    bob = HostPlacement(name="bob")
    carole = HostPlacement(name="carole")
    rep = ReplicatedPlacement(name="rep", player_names=["alice", "bob", "carole"])

    comp.add_placement(alice)
    comp.add_placement(bob)
    comp.add_placement(carole)
    comp.add_placement(rep)

    return alice, bob, carole, rep


def _compile_and_run(comp, alice, bob, carole):
    compiler = Compiler()
    comp = compiler.run_passes(comp)
    return run_test_computation(comp, [alice, bob, carole])


class ReplicatedProtocolsTest(parameterized.TestCase):
    @parameterized.parameters(
        (lambda x, y: x + y, False, standard_dialect.AddOperation),
        (lambda x, y: x - y, False, standard_dialect.SubOperation),
        (lambda x, y: x * y, False, standard_dialect.MulOperation),
        (lambda x, y: x * (x * y), True, standard_dialect.MulOperation),
    )
    @settings(max_examples=50, deadline=timedelta(milliseconds=400))
    @given(pair_lists)
    def test_bin_op(self, numpy_lmbd, consecutive_flag, replicated_std_op, bin_args):
        comp = Computation(operations={}, placements={})
        alice, bob, carole, rep = _setup_replicated_computation(comp)

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
        op_name = "rep_op"

        comp.add_operation(
            replicated_std_op(
                name="rep_op",
                placement_name=rep.name,
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                output_type=TensorType(datatype="float"),
            )
        )
        if consecutive_flag:
            comp.add_operation(
                replicated_std_op(
                    name="rep_op_2",
                    placement_name=rep.name,
                    inputs={"lhs": "alice_input", "rhs": "rep_op"},
                    output_type=TensorType(datatype="float"),
                )
            )
            op_name = "rep_op_2"

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value="result",
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": op_name},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output", placement_name=carole.name, inputs={"value": "save"},
            )
        )

        results = _compile_and_run(comp, alice, bob, carole)
        np.testing.assert_array_equal(z, results[carole]["result"])

    @parameterized.parameters([0, 1, None])
    def test_sum_op(self, axis):
        comp = Computation(operations={}, placements={})

        alice, bob, carole, rep = _setup_replicated_computation(comp)

        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        z = np.sum(x, axis=axis)

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
            standard_dialect.SumOperation(
                name="rep_op",
                placement_name=rep.name,
                axis=axis,
                inputs={"x": "alice_input"},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value="result",
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "rep_op"},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output", placement_name=carole.name, inputs={"value": "save"},
            )
        )

        results = _compile_and_run(comp, alice, bob, carole)
        np.testing.assert_array_equal(z, results[carole]["result"])

    @parameterized.parameters(
        itertools.product(
            [0, 1, None],
            [
                (np.mean, standard_dialect.MeanOperation, "mean"),
                (np.sum, standard_dialect.SumOperation, "sum"),
            ],
        )
    )
    def test_reduce_op(self, axis, reduce_op_bundle):
        np_op, std_op, op_name = reduce_op_bundle
        comp = Computation(operations={}, placements={})
        alice, bob, carole, rep = _setup_replicated_computation(comp)

        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        z = np_op(x, axis=axis)

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
            std_op(
                name=op_name,
                placement_name=rep.name,
                axis=axis,
                inputs={"x": "alice_input"},
                output_type=TensorType(datatype="float"),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value="result",
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": op_name},
                placement_name=carole.name,
            )
        )
        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output", placement_name=carole.name, inputs={"value": "save"},
            )
        )

        results = _compile_and_run(comp, alice, bob, carole)
        np.testing.assert_array_equal(z, results[carole]["result"])

    @settings(deadline=None)
    @given(dotprod_inputs)
    def test_dot_prod(self, dotprod_args):
        comp = Computation(operations={}, placements={})

        alice, bob, carole, rep = _setup_replicated_computation(comp)

        a, b = dotprod_args
        x = a.astype(np.float64)
        y = b.astype(np.float64)

        z = x @ y

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
            standard_dialect.DotOperation(
                name="dot_op",
                placement_name=rep.name,
                inputs={"lhs": "alice_input", "rhs": "bob_input"},
                output_type=TensorType(datatype="float"),
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value="result",
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "dot_op"},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output", placement_name=carole.name, inputs={"value": "save"},
            )
        )

        results = _compile_and_run(comp, alice, bob, carole)
        np.testing.assert_allclose(
            z, results[carole]["result"], rtol=1e-5, atol=1e-4,
        )

    # TODO(Dragos) insert hypothesis
    def test_abs(self):
        comp = Computation(operations={}, placements={})

        alice, bob, carole, rep = _setup_replicated_computation(comp)

        x = np.array([[1, 2], [3, 4]], dtype=np.float64)

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
            standard_dialect.AbsOperation(
                name="abs_op",
                placement_name=rep.name,
                inputs={"x": "alice_input"},
                output_type=TensorType(datatype="int"),
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"value": "abs_op"},
                placement_name=carole.name,
                key="result",
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output", placement_name=carole.name, inputs={"value": "save"},
            )
        )

        runtime = _compile_and_run(comp, alice, bob, carole)

        np.testing.assert_allclose(
            z, runtime.get_executor(carole.name).store["result"],
        )


if __name__ == "__main__":
    unittest.main()
