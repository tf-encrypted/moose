import itertools
import pathlib

import numpy as np
import onnx
from absl.testing import parameterized

import pymoose as pm
from pymoose import elk_compiler
from pymoose import runtime as rt
from pymoose.computation import utils as comp_utils
from pymoose.predictors import linear_predictor
from pymoose.predictors import predictor
from pymoose.predictors import predictor_utils

_SK_REGRESSION_MODELS = [
    ("ard_regression", [229.01876595, 52.28809679]),
    ("bayesian_ridge", [229.09049241, 52.24556655]),
    ("elastic_net", [147.18095356, 42.07852916]),
    ("elastic_net_cv", [206.35095678, 50.08868147]),
    ("huber_regressor", [230.20332624, 52.069104]),
    ("lars", [229.15556878, 52.25108639]),
    ("lars_cv", [229.15556878, 52.25108639]),
    ("lasso", [224.35101761, 52.25267607]),
    ("lasso_cv", [228.68959623, 52.25189918]),
    ("lasso_lars_ic", [229.15556878, 52.25108639]),
    ("linear_regression", [229.15556878, 52.25108639]),
    (
        "linear_regression_2targets",
        [[286.51203231, 263.60647448], [-4.26446802, 49.64789211]],
    ),
    ("orthogonal_matching_pursuit", [107.84179564, -29.4706721]),
    ("orthogonal_matching_pursuit_cv", [229.15556878, 52.25108639]),
    ("passive_aggressive_regressor", [236.36241952, 56.54396034]),
    ("quantile_regressor", [9.9146497, 9.91464969]),
    ("ransac_regressor", [229.15556878, 52.25108639]),
    ("ridge", [226.60022295, 52.03102657]),
    ("ridge_cv", [228.89720599, 52.22914582]),
    ("sgd_regressor", [229.10920079, 52.24079892]),
    ("theil_sen_regressor", [233.23775367, 51.64095974]),
]
_SK_CLASSIFIER_MODELS = [
    ("logistic_regression_2class_multiclass", [0.5535523, 0.4464477]),
    ("logistic_regression_2class_multiclass_autosk", [0.56790665, 0.43209335]),
    ("logistic_regression_3class_multiclass", [0.11341234, 0.51814332, 0.36844435]),
    ("logistic_regression_2class_multilabel", [0.56790665, 0.43209335]),
    ("logistic_regression_3class_multilabel", [0.1371954, 0.46855493, 0.39424967]),
    ("logistic_regression_cv_2class_multiclass", [0.5756927, 0.4243073]),
    ("logistic_regression_cv_3class_multiclass", [0.23125406, 0.41739519, 0.35135075]),
    ("logistic_regression_cv_2class_multilabel", [0.59146365, 0.40853635]),
    ("logistic_regression_cv_3class_multilabel", [0.11771997, 0.47229152, 0.40998851]),
]


class LinearPredictorTest(parameterized.TestCase):
    def _build_linear_predictor(self, onnx_fixture, predictor_cls, aes_predictor=False):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{onnx_fixture}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            lr_onnx = onnx.load_model(model_fixture)

        if aes_predictor:
            predictor_cls = predictor.AesWrapper(predictor_cls)

        linear_model = predictor_cls.from_onnx(lr_onnx)
        return linear_model

    def _build_prediction_logic(self, model_name, predictor_cls):
        predictor = self._build_linear_predictor(model_name, predictor_cls)

        @pm.computation
        def predictor_no_aes(x: pm.Argument(predictor.alice, dtype=pm.float64)):
            with predictor.alice:
                x_fixed = pm.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor(x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    @parameterized.parameters(*_SK_REGRESSION_MODELS)
    def test_regression_logic(self, model_name, expected):
        regressor, regressor_logic = self._build_prediction_logic(
            model_name, linear_predictor.LinearRegressor
        )
        identities = [plc.name for plc in regressor.host_placements]
        runtime = rt.LocalMooseRuntime(identities)
        input_x = np.array(
            [[1.0, 1.0, 1.0, 1.0], [-0.9, 1.3, 0.6, -0.4]], dtype=np.float64
        )
        result_dict = runtime.evaluate_computation(
            computation=regressor_logic,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.asarray(expected).reshape((2, -1))
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=5)

    @parameterized.parameters(*_SK_CLASSIFIER_MODELS)
    def test_classification_logic(self, model_name, expected):
        classifier, classifier_logic = self._build_prediction_logic(
            model_name, linear_predictor.LinearClassifier
        )
        identities = [plc.name for plc in classifier.host_placements]
        runtime = rt.LocalMooseRuntime(identities)
        input_x = np.array([[-0.9, 1.3, 0.6, -0.4]], dtype=np.float64)
        result_dict = runtime.evaluate_computation(
            computation=classifier_logic,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.array([expected])
        # TODO multiple divisions seems to lose significant amount of precision
        # (hence decimal=2 here)
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=2)

    @parameterized.parameters(
        *zip(
            map(lambda x: x[0], _SK_REGRESSION_MODELS),
            itertools.repeat(linear_predictor.LinearRegressor),
        ),
        *zip(
            map(lambda x: x[0], _SK_CLASSIFIER_MODELS),
            itertools.repeat(linear_predictor.LinearClassifier),
        ),
    )
    def test_serde(self, model_name, predictor_cls):
        aes_regressor = self._build_linear_predictor(
            model_name, predictor_cls, aes_predictor=True
        )
        aes_predictor = aes_regressor()
        traced_predictor = pm.trace(aes_predictor)
        serialized = comp_utils.serialize_computation(traced_predictor)
        logical_comp_rustref = elk_compiler.compile_computation(serialized, [])
        logical_comp_rustbytes = logical_comp_rustref.to_bytes()
        pm.MooseComputation.from_bytes(logical_comp_rustbytes)
