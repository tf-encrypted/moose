import argparse
import json
import logging
import pathlib

import numpy as np
import onnx
from absl.testing import absltest
from absl.testing import parameterized

import pymoose
from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils as comp_utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime

from . import model_utils as utils
from . import xgboost_regressor


class XGBoostReplicatedExample(parameterized.TestCase):
    def _build_forest_from_onnx(self):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / "xgboost_regressor.onnx"
        with open(fixture_path, "rb") as model_fixture:
            forest_onnx = onnx.load_model(model_fixture)
        forest_model = xgboost_regressor.XGBoostForestRegressor.from_onnx(forest_onnx)
        return forest_model

    def test_xgboost_regression_example_execute(self):
        input_x = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float64)
        root_path = pathlib.Path(__file__).parent.absolute()

        with root_path / "fixtures" / "xgboost_regressor.json" as p:
            with open(p) as f:
                forest_json = json.load(f)

        forest = xgboost_regressor.XGBoostForestRegressor.from_json(forest_json)

        forest_predict = forest.predictor_factory(
            fixedpoint_dtype=utils.DEFAULT_FIXED_DTYPE
        )
        traced_model_comp = edsl.trace(forest_predict)

        storage = {plc.name: {} for plc in forest.host_placements}
        runtime = LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in forest.host_placements}
        result_dict = runtime.evaluate_computation(
            computation=traced_model_comp,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.array([26.6495275, 88.423399], dtype=np.float64)
        np.testing.assert_almost_equal(actual_result, expected_result)

    def test_serde(self):
        forest = self._build_forest_from_onnx()
        predictor = forest.predictor_factory()
        traced = edsl.trace(predictor)
        serialized = comp_utils.serialize_computation(traced)
        logical_rustref = elk_compiler.compile_computation(serialized, [])
        logical_rustbytes = logical_rustref.to_bytes()
        pymoose.MooseComputation.from_bytes(logical_rustbytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
