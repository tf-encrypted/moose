import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.computation import types as ty
from pymoose.computation import utils
from pymoose.rust import moose_runtime

_x_owner = pm.host_placement(name="x_owner")
_y_owner = pm.host_placement(name="y_owner")
_output_owner = pm.host_placement("output_owner")


@pm.computation
def add_full_storage(
    x_key: pm.Argument(_x_owner, vtype=ty.StringType()),
    y_key: pm.Argument(_y_owner, vtype=ty.StringType()),
):
    with _x_owner:
        x = pm.load(x_key, dtype=pm.float64)
    with _y_owner:
        y = pm.load(y_key, dtype=pm.float64)
    with _output_owner:
        out = pm.add(x, y)
        res = pm.save("output", out)
    return res


@pm.computation
def add_input_storage(
    x_key: pm.Argument(_x_owner, vtype=ty.StringType()),
    y_key: pm.Argument(_y_owner, vtype=ty.StringType()),
):
    with _x_owner:
        x = pm.load(x_key, dtype=pm.float64)
    with _y_owner:
        y = pm.load(y_key, dtype=pm.float64)
    with _output_owner:
        out = pm.add(x, y)
    return out


@pm.computation
def add_output_storage(
    x: pm.Argument(_x_owner, dtype=pm.float64),
    y: pm.Argument(_y_owner, dtype=pm.float64),
):
    with _output_owner:
        out = pm.add(x, y)
        res = pm.save("output", out)
    return res


@pm.computation
def add_no_storage(
    x: pm.Argument(_x_owner, dtype=pm.float64),
    y: pm.Argument(_y_owner, dtype=pm.float64),
):
    with _output_owner:
        out = pm.add(x, y)
    return out


@pm.computation
def add_multioutput(
    x: pm.Argument(_x_owner, dtype=pm.float64),
    y: pm.Argument(_y_owner, dtype=pm.float64),
):
    with _output_owner:
        out = pm.add(x, y)
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

    def _inner_prepare_runtime(self, comp, storage_dict):
        logical_comp = pm.trace(comp)
        runtime = moose_runtime.LocalRuntime(storage_mapping=storage_dict)
        comp_bin = utils.serialize_computation(logical_comp)
        return comp_bin, runtime

    def test_full_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_full_storage, self.storage_dict
        )
        outputs = runtime.evaluate_computation(comp_bin, self.storage_args)
        assert len(outputs) == 0
        result = runtime.read_value_from_storage("output_owner", "output")
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_input_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_input_storage, self.storage_dict
        )
        result = runtime.evaluate_computation(comp_bin, self.storage_args)
        np.testing.assert_array_equal(list(result.values())[0], np.array([3.0]))

    def test_output_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_output_storage, self.empty_storage
        )
        outputs = runtime.evaluate_computation(comp_bin, self.actual_args)
        assert len(outputs) == 0
        result = runtime.read_value_from_storage("output_owner", "output")
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_no_storage(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_no_storage, self.storage_dict
        )
        result = runtime.evaluate_computation(comp_bin, self.actual_args)
        np.testing.assert_array_equal(list(result.values())[0], np.array([3.0]))

    def test_multioutput(self):
        comp_bin, runtime = self._inner_prepare_runtime(
            add_multioutput, self.storage_dict
        )
        result = runtime.evaluate_computation(comp_bin, self.actual_args)
        result = sorted(result.values())
        expected = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        for r, e in zip(result, expected):
            np.testing.assert_array_equal(r, e)

    def test_write_to_storage(self):
        runtime = moose_runtime.LocalRuntime(storage_mapping=self.empty_storage)
        x = np.array([1.0, 2.0, 3.0])
        runtime.write_value_to_storage("x_owner", "x", x)
        result = runtime.read_value_from_storage("x_owner", "x")
        np.testing.assert_array_equal(x, result)

    def test_write_wrong_identity(self):
        runtime = moose_runtime.LocalRuntime(storage_mapping=self.empty_storage)
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
