import itertools
import pathlib

import numpy as np
import onnx
from absl.testing import parameterized

import pymoose as pm
from pymoose import testing
from pymoose.computation import utils as comp_utils
from pymoose.predictors import multilayer_perceptron_predictor
from pymoose.predictors import predictor_utils

_SK_REGRESSION_MODELS = [
    ("MLP_regressor_1hidden_layers_1target_logistic", [14.61085493, 10.79791606]),
    ("MLP_regressor_2hidden_layers_1target_logistic", [11.42481646, 10.5622954]),
    ("MLP_regressor_3hidden_layers_1target_logistic", [4.27051111, 4.03313707]),
    (
        "MLP_regressor_2hidden_layers_2target_logistic",
        [[9.7588123, 10.30390732], [-0.89381576, -1.49784495]],
    ),
    (
        "MLP_regressor_2hidden_layers_2target_identity",
        [[550.57188666, 431.31144628], [57.07484159, 128.76424831]],
    ),
    ("MPL_regressor_1hidden_layers_1target_relu", [32.55903733, 16.92872095]),
]
_SK_CLASSIFIER_MODELS = [
    ("MLP_classifier_1hidden_layers_2classes_logistic", [0.60823407, 0.39176593]),
    ("MLP_classfier_3hidden_layers_2classes_identity", [0.90055265, 0.09944735]),
    ("MLP_classfier_3hidden_layers_2classes_logistic", [0.96068223, 0.03931777]),
    (
        "MLP_classfier_3hidden_layers_10classes_logistic",
        [
            0.03720068,
            0.01509579,
            0.06108879,
            0.07603149,
            0.02308053,
            0.28638066,
            0.09556369,
            0.14146137,
            0.17809053,
            0.08600645,
        ],
    ),
    (
        "MLP_classfier_1hidden_layers_10classes_logistic",
        [
            0.19517187,
            0.04481253,
            0.04494591,
            0.03637586,
            0.14826206,
            0.34719177,
            0.03231297,
            0.07976163,
            0.05626911,
            0.01489627,
        ],
    ),
    (
        "MPL_classfier_1hidden_layers_10classes_relu",
        [
            0.18669855,
            0.01198506,
            0.02292363,
            0.04383366,
            0.4636751,
            0.08422828,
            0.0207348,
            0.07604881,
            0.0196595,
            0.07021261,
        ],
    ),
]


class MLPPredictorTest(parameterized.TestCase):
    def _build_MLP_predictor(self, onnx_fixture, predictor_cls):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{onnx_fixture}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            lr_onnx = onnx.load_model(model_fixture)
        model = predictor_cls.from_onnx(lr_onnx)
        return model

    def _build_prediction_logic(self, model_name, predictor_cls):
        predictor = self._build_MLP_predictor(model_name, predictor_cls)

        @pm.computation
        def predictor_no_aes(x: pm.Argument(predictor.alice, dtype=pm.float64)):
            with predictor.alice:
                x_fixed = pm.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor.neural_predictor_fn(
                    x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE
                )
                y = predictor.post_transform(y, predictor_utils.DEFAULT_FIXED_DTYPE)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    @parameterized.parameters(*_SK_REGRESSION_MODELS)
    def test_regression_logic(self, model_name, expected):
        regressor, regressor_logic = self._build_prediction_logic(
            model_name, multilayer_perceptron_predictor.MLPRegressor
        )

        storage = {plc.name: {} for plc in regressor.host_placements}
        runtime = testing.LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in regressor.host_placements}

        input_x = np.array(
            [
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    -0.9,
                    1.3,
                    0.6,
                    -0.4,
                    0.8,
                    0.2,
                    -0.1,
                    -0.9,
                    0.5,
                    0.7,
                    -0.6,
                    0.3,
                    2.3,
                    1.5,
                    1.9,
                    -1.0,
                    1.0,
                    0.6,
                    1.4,
                    -1.2,
                ],
            ],
            dtype=np.float64,
        )
        result_dict = runtime.evaluate_computation(
            computation=regressor_logic,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.asarray(expected).reshape((2, -1))
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=1)

    @parameterized.parameters(*_SK_CLASSIFIER_MODELS)
    def test_classification_logic(self, model_name, expected):
        classifier, classifier_logic = self._build_prediction_logic(
            model_name, multilayer_perceptron_predictor.MLPClassifier
        )

        storage = {plc.name: {} for plc in classifier.host_placements}
        runtime = testing.LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in classifier.host_placements}

        input_x = np.array(
            [
                [
                    -0.9,
                    1.3,
                    0.6,
                    -0.4,
                    0.8,
                    0.2,
                    -0.1,
                    -0.9,
                    0.5,
                    0.7,
                    -0.6,
                    0.3,
                    2.3,
                    1.5,
                    1.9,
                    -1.0,
                    1.0,
                    0.6,
                    1.4,
                    -1.2,
                ]
            ],
            dtype=np.float64,
        )
        result_dict = runtime.evaluate_computation(
            computation=classifier_logic,
            role_assignment=role_assignment,
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
            itertools.repeat(multilayer_perceptron_predictor.MLPRegressor),
        ),
        *zip(
            map(lambda x: x[0], _SK_CLASSIFIER_MODELS),
            itertools.repeat(multilayer_perceptron_predictor.MLPClassifier),
        ),
    )
    def test_serde(self, model_name, predictor_cls):
        regressor = self._build_MLP_predictor(model_name, predictor_cls)
        predictor = regressor.predictor_factory()
        traced_predictor = pm.trace(predictor)
        serialized = comp_utils.serialize_computation(traced_predictor)
        logical_comp_rustref = pm.elk_compiler.compile_computation(serialized, [])
        logical_comp_rustbytes = logical_comp_rustref.to_bytes()
        pm.MooseComputation.from_bytes(logical_comp_rustbytes)
        # NOTE: could also dump to disk as follows (but we don't in the test)
        # logical_comp_rustref.to_disk(path)
        # pm.MooseComputation.from_disk(path)
