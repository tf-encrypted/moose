import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import types as ty
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


def compile_and_run(traced_slice_comp, x_arg):
    storage = {
        "alice": {},
        "carole": {},
        "bob": {"x_arg": x_arg},
    }

    runtime = LocalMooseRuntime(storage_mapping=storage)
    _ = runtime.evaluate_computation(
        computation=traced_slice_comp,
        role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
        arguments={"x_uri": "x_arg"},
    )

    x_sliced = runtime.read_value_from_storage("bob", "sliced")
    return x_sliced


class SliceExample(parameterized.TestCase):
    def _setup_comp(self, slice_spec, to_dtype):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_comp(
            x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
        ):
            with bob:
                x = edsl.load(x_uri, dtype=to_dtype)[slice_spec]
                res = (edsl.save("sliced", x),)

            return res

        return my_comp

    @parameterized.parameters(
        (
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            edsl.float32,
            (slice(1, None, None), slice(1, None, None), slice(1, None, None)),
        ),
        (
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            edsl.float64,
            (slice(None, None, None), slice(None, None, None), slice(1, None, None)),
        ),
        (
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            edsl.uint64,
            (slice(None, None, None), slice(None, None, None), slice(1, None, None)),
        ),
    )
    def test_slice_types_execute(self, x, to_dtype, slice_spec):
        comp = self._setup_comp(slice_spec, to_dtype)
        traced_slice_comp = edsl.trace(comp)

        x_arg = np.array(x, dtype=to_dtype.numpy_dtype)
        x_from_runtime = compile_and_run(traced_slice_comp, x_arg)

        expected_npy = x_arg[slice_spec]
        np.testing.assert_equal(x_from_runtime, expected_npy)

    def test_basic(self):
        def setup_basic_comp():
            bob = edsl.host_placement(name="bob")

            @edsl.computation
            def my_comp(
                x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
            ):
                with bob:
                    x = edsl.load(x_uri, dtype=edsl.float64)[1:, 1:2]
                    res = (edsl.save("sliced", x),)
                return res

            return my_comp

        comp = setup_basic_comp()
        traced_slice_comp = edsl.trace(comp)
        x_arg = np.array(
            [[1, 23.0, 321, 30.321, 321], [32.0, 321, 5, 3.0, 32.0]], dtype=np.float64
        )
        x_from_runtime = compile_and_run(traced_slice_comp, x_arg)
        np.testing.assert_equal(x_from_runtime, x_arg[1:, 1:2])

    def test_basic_colons(self):
        def setup_basic_comp():
            bob = edsl.host_placement(name="bob")

            @edsl.computation
            def my_comp(
                x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
            ):
                with bob:
                    x = edsl.load(x_uri, dtype=edsl.float64)[:, 2:4]
                    res = (edsl.save("sliced", x),)
                return res

            return my_comp

        comp = setup_basic_comp()
        traced_slice_comp = edsl.trace(comp)
        x_arg = np.array(
            [[1, 23.0, 321, 30.321, 321], [32.0, 321, 5, 3.0, 32.0]], dtype=np.float64
        )
        x_from_runtime = compile_and_run(traced_slice_comp, x_arg)
        np.testing.assert_equal(x_from_runtime, x_arg[:, 2:4])

    def test_rep_basic(self):
        def setup_basic_comp():
            alice = edsl.host_placement(name="alice")
            bob = edsl.host_placement(name="bob")
            carole = edsl.host_placement(name="carole")
            rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

            @edsl.computation
            def my_comp(
                x_uri: edsl.Argument(placement=bob, vtype=ty.StringType()),
            ):
                with bob:
                    x = edsl.load(x_uri, dtype=edsl.float64)
                    x_fixed = edsl.cast(x, dtype=edsl.fixed(8, 27))

                with rep:
                    x_sliced_rep = x_fixed[1:, 1:2]

                with bob:
                    x_sliced_host = edsl.cast(x_sliced_rep, dtype=edsl.float64)
                    res = edsl.save("sliced", x_sliced_host)
                return res

            return my_comp

        comp = setup_basic_comp()
        traced_slice_comp = edsl.trace(comp)
        x_arg = np.array(
            [[1, 23.0, 321, 30.321, 321], [32.0, 321, 5, 3.0, 32.0]], dtype=np.float64
        )
        x_from_runtime = compile_and_run(traced_slice_comp, x_arg)
        np.testing.assert_equal(x_from_runtime, x_arg[1:, 1:2])

    def test_shape_slice(self):
        alice = edsl.host_placement("alice")

        @edsl.computation
        def my_comp(x: edsl.Argument(alice, edsl.float64)):
            with alice:
                x_shape = edsl.shape(x)
                sliced_shape = x_shape[1:3]
                ones_res = edsl.ones(sliced_shape, edsl.float64)
                res = edsl.save("ones", ones_res)
            return res

        traced_slice_comp = edsl.trace(my_comp)
        x_arg = np.ones([4, 3, 5], dtype=np.float64)
        runtime = LocalMooseRuntime(storage_mapping={"alice": {}})
        _ = runtime.evaluate_computation(
            computation=traced_slice_comp,
            role_assignment={"alice": "alice"},
            arguments={"x": x_arg},
        )

        y_ones = runtime.read_value_from_storage("alice", "ones")
        assert y_ones.shape == (3, 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="comparison example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
