import pathlib

import numpy as np
import onnx
from absl.testing import parameterized

import pymoose
from pymoose import edsl
from pymoose import elk_compiler
from pymoose import testing
from pymoose.computation import utils as comp_utils

from . import linear_predictor
from . import model_utils

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
    # TODO uncomment when softmax implemented in LinearClassifier
    # "logistic_regression_2class_multiclass",
    # "logistic_regression_3class_multiclass",
    "logistic_regression_2class_multilabel",
    "logistic_regression_3class_multilabel",
]


class LinearPredictorTest(parameterized.TestCase):
    def _build_linear_predictor(self, onnx_fixture, predictor_cls):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{onnx_fixture}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            lr_onnx = onnx.load_model(model_fixture)
        linear_model = predictor_cls.from_onnx(lr_onnx)
        return linear_model

    @parameterized.parameters(*_SK_REGRESSION_MODELS)
    def test_regression_logic(self, model_name):
        regressor = self._build_linear_predictor(
            model_name, linear_predictor.LinearRegressor
        )

        @edsl.computation
        def predictor_minus_aes(x: edsl.Argument(regressor.alice, dtype=edsl.float64)):
            with regressor.alice:
                x_fixed = edsl.cast(x, dtype=model_utils.DEFAULT_FIXED_DTYPE)
            y = regressor.linear_predictor_fn(x_fixed, model_utils.DEFAULT_FIXED_DTYPE)
            return model_utils.handle_predictor_output(
                y, prediction_handler=regressor.bob
            )

        traced_predictor = edsl.trace(predictor_minus_aes)
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
        expected_result = regressor.coeffs.sum() + regressor.intercepts
        np.testing.assert_almost_equal(actual_result, expected_result)

    @parameterized.parameters(*_SK_REGRESSION_MODELS)
    def test_serde(self, model_name):
        linear_model = self._build_linear_predictor(model_name)
        predictor = linear_model.predictor_factory()
        traced_predictor = edsl.trace(predictor)
        serialized = comp_utils.serialize_computation(traced_predictor)
        logical_comp_rustref = elk_compiler.compile_computation(serialized, [])
        logical_comp_rustbytes = logical_comp_rustref.to_bytes()
        pymoose.MooseComputation.from_bytes(logical_comp_rustbytes)
        # NOTE: could also dump to disk as follows (but we don't in the test)
        # logical_comp_rustref.to_disk(path)
        # pymoose.MooseComputation.from_disk(path)
