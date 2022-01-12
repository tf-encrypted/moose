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

from . import model_utils
from . import tree_ensemble_regressor


class TreeEnsembleRegressorTest(parameterized.TestCase):
    def _build_forest_from_onnx(self):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / "xgboost_regressor.onnx"
        with open(fixture_path, "rb") as model_fixture:
            forest_onnx = onnx.load_model(model_fixture)
        forest_model = tree_ensemble_regressor.TreeEnsembleRegressor.from_onnx(
            forest_onnx
        )
        return forest_model

    def _build_forest_from_json(self):
        root_path = pathlib.Path(__file__).parent.absolute()
        with root_path / "fixtures" / "xgboost_regressor.json" as p:
            with open(p) as f:
                forest_json = json.load(f)
        forest_model = tree_ensemble_regressor.TreeEnsembleRegressor.from_json(
            forest_json
        )
        return forest_model

    def _build_prediction_logic(self, onnx_or_json):
        if onnx_or_json == "onnx":
            predictor = self._build_forest_from_onnx()
        elif onnx_or_json == "json":
            predictor = self._build_forest_from_json()
        else:
            raise ValueError()

        @edsl.computation
        def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
            with predictor.alice:
                x_fixed = edsl.cast(x, dtype=model_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor._forest_fn(x_fixed, model_utils.DEFAULT_FIXED_DTYPE)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    def test_tree_ensemble_regressor_logic(self):
        input_x = np.array([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.float64)
        regressor, regression_logic = self._build_prediction_logic("onnx")
        traced_model_comp = edsl.trace(regression_logic)
        storage = {plc.name: {} for plc in regressor.host_placements}
        runtime = LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in regressor.host_placements}
        result_dict = runtime.evaluate_computation(
            computation=traced_model_comp,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.array([5.327446, 54.89666], dtype=np.float64)
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=5)

    def test_serde(self):
        forest = self._build_forest_from_onnx()
        predictor = forest.predictor_factory()
        traced = edsl.trace(predictor)
        serialized = comp_utils.serialize_computation(traced)
        logical_rustref = elk_compiler.compile_computation(serialized, [])
        logical_rustbytes = logical_rustref.to_bytes()
        pymoose.MooseComputation.from_bytes(logical_rustbytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree Ensemble predictions")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
