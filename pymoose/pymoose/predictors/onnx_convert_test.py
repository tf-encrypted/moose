import argparse
import logging
import pathlib

import onnx
from absl.testing import absltest
from absl.testing import parameterized

from pymoose.logger import get_logger
from pymoose.predictors import linear_predictor
from pymoose.predictors import neural_net_predictor
from pymoose.predictors import onnx_convert
from pymoose.predictors import tree_ensemble

_SK_MODELS = [
    ("linear_regression", linear_predictor.LinearRegressor),
    ("logistic_regression_2class_multiclass", linear_predictor.LinearClassifier),
    ("random_forest_regressor", tree_ensemble.TreeEnsembleRegressor),
    ("random_forest_classifier_2class", tree_ensemble.TreeEnsembleClassifier),
    ("xgboost_regressor", tree_ensemble.TreeEnsembleRegressor),
    ("xgboost_classifier_2class", tree_ensemble.TreeEnsembleClassifier),
    (
        "MPL_regressor_2hidden_layers_1target_logistic",
        neural_net_predictor.MLPRegressor,
    ),
    (
        "MPL_classfier_3hidden_layers_2classes_identity",
        neural_net_predictor.MLPClassifier,
    ),
]


class PredictoOnnxTest(parameterized.TestCase):
    def _load_onnx(self, onnx_fixture):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{onnx_fixture}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            onnx_proto = onnx.load_model(model_fixture)
        return onnx_proto

    @parameterized.parameters(*_SK_MODELS)
    def test_regression_logic(self, model_name, predictor_cls):
        model_proto = self._load_onnx(model_name)
        model = onnx_convert.from_onnx(model_proto)
        assert isinstance(model, predictor_cls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
