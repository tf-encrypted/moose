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
from pymoose.predictors import predictor_utils
from pymoose.predictors import tree_ensemble_regressor
from pymoose.testing import LocalMooseRuntime

_XGB_MODELS = [("xgboost_regressor", [14.121551, 14.121551, 113.279236])]
_SK_MODELS = [
    ("extra_trees_regressor", [-13.39700383, -13.39700383, 114.41578554]),
    ("random_forest_regressor", [-4.89890751, -4.89890751, 125.78084158]),
    ("gradient_boosting_regressor", [6.98515914, 0.94996615, 22.03610848]),
    ("hist_gradient_boosting_regressor", [-1.01535751, -1.01535751, 12.34103961]),
]


class TreeEnsembleRegressorTest(parameterized.TestCase):
    def _build_forest_from_onnx(self, model_name):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{model_name}.onnx"
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

    def _build_prediction_logic(self, model_name, onnx_or_json):
        if onnx_or_json == "onnx":
            predictor = self._build_forest_from_onnx(model_name)
        elif onnx_or_json == "json":
            predictor = self._build_forest_from_json()
        else:
            raise ValueError()

        @edsl.computation
        def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
            with predictor.alice:
                x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor._forest_fn(x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    @parameterized.parameters(*_XGB_MODELS + _SK_MODELS)
    def test_tree_ensemble_regressor_logic(self, model_name, expected):
        input_x = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0.2, 4, 2, 6]], dtype=np.float64
        )
        regressor, regression_logic = self._build_prediction_logic(model_name, "onnx")
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
        expected_result = np.array(expected, dtype=np.float64)
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=6)

    @parameterized.parameters(*map(lambda x: x[0], _XGB_MODELS + _SK_MODELS))
    def test_serde(self, model_name):
        forest = self._build_forest_from_onnx(model_name)
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
