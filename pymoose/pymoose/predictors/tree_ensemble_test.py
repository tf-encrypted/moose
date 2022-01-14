import argparse
import itertools
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
from pymoose.predictors import tree_ensemble
from pymoose.testing import LocalMooseRuntime

_XGB_REGRESSOR_MODELS = [("xgboost_regressor", [14.121551, 14.121551, 113.279236])]
_SK_REGRESSOR_MODELS = [
    ("extra_trees_regressor", [-13.39700383, -13.39700383, 114.41578554]),
    ("random_forest_regressor", [-4.89890751, -4.89890751, 125.78084158]),
    ("gradient_boosting_regressor", [6.98515914, 0.94996615, 22.03610848]),
    ("hist_gradient_boosting_regressor", [-1.01535751, -1.01535751, 12.34103961]),
]
_XGB_CLASSIFIER_MODELS = [
    (
        "xgboost_classifier_2class",
        [0.35881676, 0.1355824, 0.35881676],
        # TODO see TreeEnsembleClassifier._double_sigmoid in tree_ensemble module
        # [[0.64118324, 0.35881676], [0.8644176, 0.1355824], [0.64118324, 0.35881676]],
        [0.35881676, 0.1355824, 0.35881676],
    ),
    (
        "xgboost_classifier_3class",
        [
            [0.04490882, 0.91165735, 0.04343383],
            [0.04490882, 0.91165735, 0.04343383],
            [0.2080709, 0.1956092, 0.59631991],
        ],
    ),
]
_REGRESSOR_MODELS = _XGB_REGRESSOR_MODELS + _SK_REGRESSOR_MODELS
_CLASSIFIER_MODELS = _XGB_CLASSIFIER_MODELS


class TreeEnsembleRegressorTest(parameterized.TestCase):
    def _build_forest_from_onnx(self, model_name, predictor_cls):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{model_name}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            forest_onnx = onnx.load_model(model_fixture)
        forest_model = predictor_cls.from_onnx(forest_onnx)
        return forest_model

    def _build_forest_from_json(self, predictor_cls):
        root_path = pathlib.Path(__file__).parent.absolute()
        with root_path / "fixtures" / "xgboost_regressor.json" as p:
            with open(p) as f:
                forest_json = json.load(f)
        forest_model = tree_ensemble.TreeEnsembleRegressor.from_json(forest_json)
        return forest_model

    def _build_prediction_logic(self, model_name, onnx_or_json, predictor_cls):
        if onnx_or_json == "onnx":
            predictor = self._build_forest_from_onnx(model_name, predictor_cls)
        elif onnx_or_json == "json":
            predictor = self._build_forest_from_json()
        else:
            raise ValueError()

        @edsl.computation
        def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
            with predictor.alice:
                x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor.forest_fn(x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE)
                y = predictor.post_transform(y, predictor_utils.DEFAULT_FIXED_DTYPE)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    @parameterized.parameters(
        *zip(_REGRESSOR_MODELS, itertools.repeat(tree_ensemble.TreeEnsembleRegressor),),
        # *zip(
        #     _CLASSIFIER_MODELS,
        #     itertools.repeat(tree_ensemble.TreeEnsembleClassifier),
        # ),
    )
    def test_tree_ensemble_logic(self, test_case, predictor_cls):
        model_name, expected = test_case
        input_x = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0.2, 4, 2, 6]], dtype=np.float64
        )
        predictor, predictor_logic = self._build_prediction_logic(
            model_name, "onnx", predictor_cls
        )
        traced_model_comp = edsl.trace(predictor_logic)
        storage = {plc.name: {} for plc in predictor.host_placements}
        runtime = LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in predictor.host_placements}
        result_dict = runtime.evaluate_computation(
            computation=traced_model_comp,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.array(expected, dtype=np.float64)
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=6)

    @parameterized.parameters(
        *zip(
            map(lambda x: x[0], _REGRESSOR_MODELS),
            itertools.repeat(tree_ensemble.TreeEnsembleRegressor),
        ),
        *zip(
            map(lambda x: x[0], _CLASSIFIER_MODELS),
            itertools.repeat(tree_ensemble.TreeEnsembleClassifier),
        ),
    )
    def test_serde(self, model_name, predictor_cls):
        forest = self._build_forest_from_onnx(model_name, predictor_cls)
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
