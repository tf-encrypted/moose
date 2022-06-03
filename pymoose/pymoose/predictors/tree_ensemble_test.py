import argparse
import itertools
import json
import logging
import pathlib

import numpy as np
import onnx
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose import runtime as rt
from pymoose.logger import get_logger
from pymoose.predictors import predictor_utils
from pymoose.predictors import tree_ensemble

_XGB_REGRESSOR_MODELS = [("xgboost_regressor", [14.121551, 14.121551, 113.279236])]
_SK_REGRESSOR_MODELS = [
    ("extra_trees_regressor", [-13.39700383, -13.39700383, 114.41578554]),
    ("gradient_boosting_regressor", [6.98515914, 0.94996615, 22.03610848]),
    ("hist_gradient_boosting_regressor", [-1.01535751, -1.01535751, 12.34103961]),
    ("random_forest_regressor", [-4.89890751, -4.89890751, 125.78084158]),
]
_XGB_CLASSIFIER_MODELS = [
    (
        "xgboost_classifier_2class",
        [[0.64118324, 0.35881676], [0.8644176, 0.1355824], [0.64118326, 0.35881677]],
    ),
    (
        "xgboost_classifier_2class_5trees",
        [[0.8271516, 0.17284839], [0.9636785, 0.03632155], [0.9332141, 0.06678587]],
    ),
    (
        "xgboost_classifier_3class",
        [
            [0.04490882, 0.91165735, 0.04343383],
            [0.04490882, 0.91165735, 0.04343383],
            [0.2080709, 0.1956092, 0.59631991],
        ],
    ),
    (
        "xgboost_classifier_3class_5trees",
        [
            [0.0064704, 0.988167, 0.00536254],
            [0.00647771, 0.98928285, 0.00423949],
            [0.09881791, 0.40364704, 0.49753505],
        ],
    ),
]
_SK_CLASSIFIER_MODELS = [
    (
        "extra_trees_classifier_2class",
        [[0.27536232, 0.72463768], [0.39247312, 0.60752688], [1.0, 0.0]],
    ),
    (
        "extra_trees_classifier_3class",
        [
            [0.19040698, 0.75145349, 0.05813953],
            [0.19040698, 0.75145349, 0.05813953],
            [0.29457364, 0.31395349, 0.39147287],
        ],
    ),
    (
        "gradient_boosting_classifier_2class",
        [[0.54741548, 0.45258452], [0.54741548, 0.45258452], [0.54741548, 0.45258452]],
    ),
    (
        "gradient_boosting_classifier_3class",
        [
            [0.27390542, 0.46277432, 0.26332026],
            [0.27390542, 0.46277432, 0.26332026],
            [0.27905818, 0.26810226, 0.45283956],
        ],
    ),
    (
        "hist_gradient_boosting_classifier_2class",
        [[0.54566526, 0.45433474], [0.54566526, 0.45433474], [0.54566526, 0.45433474]],
    ),
    (
        "hist_gradient_boosting_classifier_3class",
        [
            [0.25403482, 0.50442912, 0.24153607],
            [0.24696206, 0.49038495, 0.26265299],
            [0.26277792, 0.25715092, 0.48007117],
        ],
    ),
    (
        "random_forest_classifier_2class",
        [[0.3, 0.7], [0.7047619, 0.2952381], [0.5, 0.5]],
    ),
    (
        "random_forest_classifier_2class_5trees",
        [[0.50857143, 0.49142857], [0.72857143, 0.27142857], [0.64666667, 0.35333333]],
    ),
    (
        "random_forest_classifier_3class",
        [[0.03571429, 0.9642857, 0.0], [0.03571429, 0.9642857, 0.0], [0.5, 0.5, 0.0]],
    ),
    (
        "random_forest_classifier_3class_5trees",
        [
            [0.01428571, 0.98571429, 0.0],
            [0.01428571, 0.98571429, 0.0],
            [0.33333333, 0.6, 0.06666667],
        ],
    ),
]
_REGRESSOR_MODELS = _SK_REGRESSOR_MODELS + _XGB_REGRESSOR_MODELS
_CLASSIFIER_MODELS = _SK_CLASSIFIER_MODELS + _XGB_CLASSIFIER_MODELS


class TreeEnsembleTest(parameterized.TestCase):
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

        @pm.computation
        def predictor_no_aes(x: pm.Argument(predictor.alice, dtype=pm.float64)):
            with predictor.alice:
                x_fixed = pm.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor(x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    @parameterized.parameters(
        *zip(_REGRESSOR_MODELS, itertools.repeat(tree_ensemble.TreeEnsembleRegressor)),
        *zip(
            _CLASSIFIER_MODELS,
            itertools.repeat(tree_ensemble.TreeEnsembleClassifier),
        ),
    )
    def test_tree_ensemble_logic(self, test_case, predictor_cls):
        model_name, expected = test_case
        input_x = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [0.2, 4, 2, 6]], dtype=np.float64
        )
        predictor, predictor_logic = self._build_prediction_logic(
            model_name, "onnx", predictor_cls
        )
        identities = [plc.name for plc in predictor.host_placements]
        runtime = rt.LocalMooseRuntime(identities)
        result_dict = runtime.evaluate_computation(
            computation=predictor_logic,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.array(expected, dtype=np.float64)
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree Ensemble predictions")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
