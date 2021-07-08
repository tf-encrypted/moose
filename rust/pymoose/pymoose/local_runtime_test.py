import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from moose import edsl
from moose.computation.standard import StringType
from moose.computation.utils import serialize_computation

from pymoose import LocalRuntime

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
        res = edsl.save("output", out)
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
    return (out, x, y)


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
        self.storage_args = {"x_key": "x", "y_key": "y"}
        self.actual_args = {**self.x_input, **self.y_input}
        self.role_assignment = {
            "x_owner": "x_owner",
            "y_owner": "y_owner",
            "output_owner": "output_owner",
        }

    def _inner_prepare_runtime(self, comp, storage_dict):
        concrete_comp = edsl.trace_and_compile(comp, ring=128)
        comp_bin = serialize_computation(concrete_comp)
        runtime = LocalRuntime(storage_mapping=storage_dict)
        return comp_bin, runtime

    def test_full_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_full_storage, self.storage_dict
        )
        outputs = runtime.evaluate_computation(
            comp_bin, self.role_assignment, self.storage_args
        )
        assert len(outputs) == 0
        result = runtime.read_value_from_storage("output_owner", "output")
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_input_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_input_storage, self.storage_dict
        )
        result = runtime.evaluate_computation(
            comp_bin, self.role_assignment, self.storage_args
        )
        np.testing.assert_array_equal(result["output_0"], np.array([3.0]))

    def test_output_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_output_storage, self.empty_storage
        )
        outputs = runtime.evaluate_computation(
            comp_bin, self.role_assignment, self.actual_args
        )
        assert len(outputs) == 0
        result = runtime.read_value_from_storage("output_owner", "output")
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_no_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_no_storage, self.storage_dict
        )
        result = runtime.evaluate_computation(
            comp_bin, self.role_assignment, self.actual_args
        )
        np.testing.assert_array_equal(result["output_0"], np.array([3.0]))

    def test_multioutput(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_multioutput, self.storage_dict
        )
        result = runtime.evaluate_computation(
            comp_bin, self.role_assignment, self.actual_args
        )
        np.testing.assert_array_equal(result["output_0"], np.array([3.0]))
        np.testing.assert_array_equal(result["output_1"], np.array([1.0]))
        np.testing.assert_array_equal(result["output_2"], np.array([2.0]))

    def test_write_to_storage(self):
        runtime = LocalRuntime(storage_mapping=self.empty_storage)
        x = np.array([1.0, 2.0, 3.0])
        runtime.write_value_to_storage("x_owner", "x", x)
        result = runtime.read_value_from_storage("x_owner", "x")
        np.testing.assert_array_equal(x, result)

    def test_write_wrong_identity(self):
        runtime = LocalRuntime(storage_mapping=self.empty_storage)
        x = np.array([1.0, 2.0, 3.0])
        self.assertRaises(
            RuntimeError,
            runtime.write_value_to_storage,
            identity="missingidentity",
            key="x",
            value=x,
        )


if __name__ == "__main__":
    absltest.main()
