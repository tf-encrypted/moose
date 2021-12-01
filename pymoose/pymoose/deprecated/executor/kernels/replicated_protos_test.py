import itertools
import unittest
from datetime import timedelta

import numpy as np
from absl.testing import parameterized
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pymoose.computation import dtypes
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import StringConstant
from pymoose.computation.standard import TensorConstant
from pymoose.computation.standard import TensorType
from pymoose.computation.standard import UnitType
from pymoose.deprecated.compiler.compiler import Compiler
from pymoose.deprecated.computation import fixedpoint as fixedpoint_ops
from pymoose.deprecated.testing import run_test_computation

# to get 2 random lists of equal size using hypothesis
# https://stackoverflow.com/questions/51597021/python-hypothesis-ensure-that-input-lists-have-same-length

pair_lists = st.lists(
    st.tuples(
        st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=8)
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


def compile(comp, alice, bob, carole):
    compiler = Compiler()
    comp = compiler.run_passes(comp)
    return comp


def run(comp, alice, bob, carole):
    return run_test_computation(comp, [alice, bob, carole])


class ReplicatedProtocolsTest(parameterized.TestCase):
    @parameterized.parameters(
        (lambda x, y: x * y, standard_dialect.MulOperation, 28),
        (lambda x, y: x * y, standard_dialect.MulOperation, 29),
        (lambda x, y: x * y, standard_dialect.MulOperation, 30),
        (lambda x, y: x * y, standard_dialect.MulOperation, 31),
    )
    def test_trunc(self, numpy_lmbd, replicated_std_op, frac_precision):
        comp = Computation(operations={}, placements={})
        alice, bob, carole, rep = _setup_replicated_computation(comp)

        x = np.array([-1], dtype=np.float64)
        y = np.array([1], dtype=np.float64)

        z = numpy_lmbd(x, y)

        fp_dtype = dtypes.fixed(8, frac_precision)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=TensorConstant(value=x),
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="bob_input",
                value=TensorConstant(value=y),
                placement_name=bob.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_bob",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        op_name = "rep_op"

        comp.add_operation(
            replicated_std_op(
                name="rep_op",
                placement_name=rep.name,
                inputs={"lhs": "encode_alice", "rhs": "encode_bob"},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value=StringConstant(value="result"),
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_carole",
                inputs={"value": op_name},
                placement_name=carole.name,
                output_type=TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "decode_carole"},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output",
                placement_name=carole.name,
                inputs={"value": "save"},
                output_type=UnitType(),
            )
        )

        compiled_computation = compile(comp, alice, bob, carole)
        for i in range(100):
            results = run(compiled_computation, alice, bob, carole)
            np.testing.assert_array_equal(z, results[carole]["result"])

    @parameterized.parameters(
        (lambda x, y: x + y, False, standard_dialect.AddOperation),
        (lambda x, y: x - y, False, standard_dialect.SubOperation),
        (lambda x, y: x * y, False, standard_dialect.MulOperation),
        (lambda x, y: x * (x * y), True, standard_dialect.MulOperation),
    )
    @settings(deadline=timedelta(milliseconds=400))
    @given(pair_lists)
    def test_bin_op(self, numpy_lmbd, consecutive_flag, replicated_std_op, bin_args):
        comp = Computation(operations={}, placements={})
        alice, bob, carole, rep = _setup_replicated_computation(comp)

        a, b = map(list, zip(*bin_args))
        x = np.array(a, dtype=np.float64)
        y = np.array(b, dtype=np.float64)

        z = numpy_lmbd(x, y)

        fp_dtype = dtypes.fixed(8, 27)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=TensorConstant(value=x),
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="bob_input",
                value=TensorConstant(value=y),
                placement_name=bob.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_bob",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        op_name = "rep_op"

        comp.add_operation(
            replicated_std_op(
                name="rep_op",
                placement_name=rep.name,
                inputs={"lhs": "encode_alice", "rhs": "encode_bob"},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        if consecutive_flag:
            comp.add_operation(
                replicated_std_op(
                    name="rep_op_2",
                    placement_name=rep.name,
                    inputs={"lhs": "encode_alice", "rhs": "rep_op"},
                    output_type=TensorType(dtype=dtypes.float64),
                )
            )
            op_name = "rep_op_2"

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value=StringConstant(value="result"),
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_carole",
                inputs={"value": op_name},
                placement_name=carole.name,
                output_type=TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "decode_carole"},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output",
                placement_name=carole.name,
                inputs={"value": "save"},
                output_type=UnitType(),
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

        fp_dtype = dtypes.fixed(8, 27)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=TensorConstant(value=x),
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.SumOperation(
                name="rep_op",
                placement_name=rep.name,
                axis=axis,
                inputs={"x": "encode_alice"},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value=StringConstant(value="result"),
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_carole",
                inputs={"value": "rep_op"},
                placement_name=carole.name,
                output_type=TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "decode_carole"},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output",
                placement_name=carole.name,
                inputs={"value": "save"},
                output_type=UnitType(),
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

        fp_dtype = dtypes.fixed(8, 27)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=TensorConstant(value=x),
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            std_op(
                name=op_name,
                placement_name=rep.name,
                axis=axis,
                inputs={"x": "encode_alice"},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value=StringConstant(value="result"),
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_carole",
                inputs={"value": op_name},
                placement_name=carole.name,
                output_type=TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "decode_carole"},
                placement_name=carole.name,
            )
        )
        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output",
                placement_name=carole.name,
                inputs={"value": "save"},
                output_type=UnitType(),
            )
        )

        results = _compile_and_run(comp, alice, bob, carole)
        np.testing.assert_array_equal(z, results[carole]["result"])

    @settings(deadline=timedelta(milliseconds=400))
    @given(dotprod_inputs)
    def test_dot_prod(self, dotprod_args):
        comp = Computation(operations={}, placements={})

        alice, bob, carole, rep = _setup_replicated_computation(comp)

        a, b = dotprod_args
        x = a.astype(np.float64)
        y = b.astype(np.float64)

        z = x @ y

        fp_dtype = dtypes.fixed(8, 27)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=TensorConstant(value=x),
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="bob_input",
                value=TensorConstant(value=y),
                placement_name=bob.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_bob",
                inputs={"value": "bob_input"},
                placement_name="bob",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.DotOperation(
                name="dot_op",
                placement_name=rep.name,
                inputs={"lhs": "encode_alice", "rhs": "encode_bob"},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value=StringConstant(value="result"),
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_carole",
                inputs={"value": "dot_op"},
                placement_name=carole.name,
                output_type=TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "decode_carole"},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output",
                placement_name=carole.name,
                inputs={"value": "save"},
                output_type=UnitType(),
            )
        )

        results = _compile_and_run(comp, alice, bob, carole)
        np.testing.assert_allclose(
            z, results[carole]["result"], rtol=1e-5, atol=1e-4,
        )

    @settings(deadline=timedelta(milliseconds=2000))
    @given(a=st.lists(st.integers(min_value=-4000, max_value=4000), min_size=1))
    def test_abs(self, a):
        comp = Computation(operations={}, placements={})

        alice, bob, carole, rep = _setup_replicated_computation(comp)

        x = np.array([a], dtype=np.float64)

        fp_dtype = dtypes.fixed(8, 27)

        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="alice_input",
                value=TensorConstant(value=x),
                placement_name=alice.name,
                inputs={},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )

        comp.add_operation(
            fixedpoint_ops.EncodeOperation(
                name="encode_alice",
                inputs={"value": "alice_input"},
                placement_name="alice",
                output_type=fixedpoint_ops.EncodedTensorType(
                    dtype=fp_dtype, precision=fp_dtype.fractional_precision
                ),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.AbsOperation(
                name="abs_op",
                placement_name=rep.name,
                inputs={"x": "encode_alice"},
                output_type=TensorType(dtype=dtypes.float64),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                inputs={},
                placement_name=carole.name,
                value=StringConstant(value="result"),
                output_type=standard_dialect.StringType(),
            )
        )

        comp.add_operation(
            fixedpoint_ops.DecodeOperation(
                name="decode_result",
                inputs={"value": "abs_op"},
                placement_name=carole.name,
                output_type=TensorType(dtype=dtypes.float64),
                precision=fp_dtype.fractional_precision,
            )
        )

        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                inputs={"key": "save_key", "value": "decode_result"},
                placement_name=carole.name,
            )
        )

        comp.add_operation(
            standard_dialect.OutputOperation(
                name="output",
                placement_name=carole.name,
                inputs={"value": "save"},
                output_type=UnitType(),
            )
        )

        results = _compile_and_run(comp, alice, bob, carole)

        np.testing.assert_allclose(
            np.abs(x), results[carole]["result"],
        )


if __name__ == "__main__":
    unittest.main()