import argparse
import logging
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import onnx as ox

from pymoose.logger import get_logger
from pymoose.predictors import linear_predictor
from pymoose.predictors import transcribe_onnx
from pymoose.predictors import tree_ensemble
from pymoose import edsl

_SK_MODELS = [
    ("linear_regression", linear_predictor.LinearRegressor),
    ("logistic_regression_2class_multiclass", linear_predictor.LinearClassifier),
    ("random_forest_regressor", tree_ensemble.TreeEnsembleRegressor),
    ("random_forest_classifier_2class", tree_ensemble.TreeEnsembleClassifier),
    ("xgboost_regressor", tree_ensemble.TreeEnsembleRegressor),
    ("xgboost_classifier_2class", tree_ensemble.TreeEnsembleClassifier),
]


class PredictoOnnxTest(parameterized.TestCase):
    def _load_onnx(self, onnx_fixture):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{onnx_fixture}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            onnx_proto = ox.load_model(model_fixture)
        return onnx_proto

    @parameterized.parameters(*_SK_MODELS)
    def test_regression_logic(self, model_name, predictor_cls):
        model_proto = self._load_onnx(model_name)
        model = transcribe_onnx.from_onnx(model_proto)
        assert isinstance(model, predictor_cls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
