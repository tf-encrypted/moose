import itertools
import pathlib

import numpy as np
import onnx
from absl.testing import parameterized

import pymoose
from pymoose import edsl
from pymoose import elk_compiler
from pymoose import testing
from pymoose.computation import utils as comp_utils
from pymoose.predictors import linear_predictor
from pymoose.predictors import predictor_utils

_SK_REGRESSION_MODELS = [
    "ard_regression",
    "bayesian_ridge",
    "elastic_net",
    "elastic_net_cv",
    "huber_regressor",
    "lars",
    "lars_cv",
    "lasso",
    "lasso_cv",
    "lasso_lars_ic",
    "linear_regression",
    "linear_regression_2targets",
    "orthogonal_matching_pursuit",
    "orthogonal_matching_pursuit_cv",
    "passive_aggressive_regressor",
    "quantile_regressor",
    "ransac_regressor",
    "ridge",
    "ridge_cv",
    "sgd_regressor",
    "theil_sen_regressor",
]
_SK_CLASSIFIER_MODELS = [
    ("logistic_regression_2class_multiclass", [0.5535523, 0.4464477]),
    ("logistic_regression_3class_multiclass", [0.11341234, 0.51814332, 0.36844435]),
    ("logistic_regression_2class_multilabel", [0.56790665, 0.43209335]),
    ("logistic_regression_3class_multilabel", [0.1371954, 0.46855493, 0.39424967]),
    ("logistic_regression_cv_2class_multiclass", [0.5756927, 0.4243073]),
    ("logistic_regression_cv_3class_multiclass", [0.23125406, 0.41739519, 0.35135075]),
    ("logistic_regression_cv_2class_multilabel", [0.59146365, 0.40853635]),
    ("logistic_regression_cv_3class_multilabel", [0.11771997, 0.47229152, 0.40998851]),
]


class LinearPredictorTest(parameterized.TestCase):
    def _build_linear_predictor(self, onnx_fixture, predictor_cls):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{onnx_fixture}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            lr_onnx = onnx.load_model(model_fixture)
        linear_model = predictor_cls.from_onnx(lr_onnx)
        return linear_model

    def _build_prediction_logic(self, model_name, predictor_cls):
        predictor = self._build_linear_predictor(model_name, predictor_cls)

        @edsl.computation
        def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
            with predictor.alice:
                x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor.linear_predictor_fn(
                    x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE
                )
                y = predictor.post_transform(y)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    @parameterized.parameters(*_SK_REGRESSION_MODELS)
    def test_regression_logic(self, model_name):
        print("Model name:", model_name)
        regressor, regressor_logic = self._build_prediction_logic(
            model_name, linear_predictor.LinearRegressor
        )

        traced_predictor = edsl.trace(regressor_logic)
        storage = {plc.name: {} for plc in regressor.host_placements}
        runtime = testing.LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in regressor.host_placements}

        input_x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        result_dict = runtime.evaluate_computation(
            computation=traced_predictor,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        # predicting on input vector of all ones == sum of linear model's coefficients
        expected_result = regressor.coeffs.sum() + regressor.intercepts.sum()
        np.testing.assert_almost_equal(actual_result.sum(), expected_result)

    @parameterized.parameters(*_SK_CLASSIFIER_MODELS)
    def test_classification_logic(self, model_name, expected):
        classifier, classifier_logic = self._build_prediction_logic(
            model_name, linear_predictor.LinearClassifier
        )

        traced_predictor = edsl.trace(classifier_logic)
        storage = {plc.name: {} for plc in classifier.host_placements}
        runtime = testing.LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in classifier.host_placements}

        input_x = np.array([[-0.9, 1.3, 0.6, -0.4]], dtype=np.float64)
        result_dict = runtime.evaluate_computation(
            computation=traced_predictor,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.array([expected])
        # TODO multiple divisions seems to lose significant amount of precision
        # (hence decimal=2 here)
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=2)

    @parameterized.parameters(
        *zip(_SK_REGRESSION_MODELS, itertools.repeat(linear_predictor.LinearRegressor)),
        *zip(
            map(lambda x: x[0], _SK_CLASSIFIER_MODELS),
            itertools.repeat(linear_predictor.LinearClassifier),
        ),
    )
    def test_serde(self, model_name, predictor_cls):
        regressor = self._build_linear_predictor(model_name, predictor_cls)
        predictor = regressor.predictor_factory()
        traced_predictor = edsl.trace(predictor)
        serialized = comp_utils.serialize_computation(traced_predictor)
        logical_comp_rustref = elk_compiler.compile_computation(serialized, [])
        logical_comp_rustbytes = logical_comp_rustref.to_bytes()
        pymoose.MooseComputation.from_bytes(logical_comp_rustbytes)
        # NOTE: could also dump to disk as follows (but we don't in the test)
        # logical_comp_rustref.to_disk(path)
        # pymoose.MooseComputation.from_disk(path)
