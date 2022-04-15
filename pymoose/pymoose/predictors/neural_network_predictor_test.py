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
from pymoose.predictors import neural_network_predictor
from pymoose.predictors import predictor_utils

_MODELS = [
    ("pytorch_net_1hidden_layer_sigmoid", [[0.32620516, 0.6737948]]),
    ("pytorch_net_2hidden_layer_sigmoid", [[0.5311462, 0.46885377]]),
    ("keras_net_1hidden_layer_sigmoid", [[0.7901215, 0.20987843]]),
    ("keras_net_2hidden_layer_sigmoid", [[0.37569475, 0.6243052]]),
    ("pytorch_net_1hidden_layer_relu", [[0.53269005, 0.46730995]]),
]


class NNPredictorTest(parameterized.TestCase):
    def _build_NN_predictor(self, onnx_fixture, predictor_cls):
        root_path = pathlib.Path(__file__).parent.absolute()
        fixture_path = root_path / "fixtures" / f"{onnx_fixture}.onnx"
        with open(fixture_path, "rb") as model_fixture:
            lr_onnx = onnx.load_model(model_fixture)
        model = predictor_cls.from_onnx(lr_onnx)
        return model

    def _build_prediction_logic(self, model_name, predictor_cls):
        predictor = self._build_NN_predictor(model_name, predictor_cls)

        @edsl.computation
        def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
            with predictor.alice:
                x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            with predictor.replicated:
                y = predictor.neural_predictor_fn(
                    x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE
                )
            return predictor.handle_output(y, prediction_handler=predictor.bob)

        return predictor, predictor_no_aes

    @parameterized.parameters(*_MODELS)
    def test_regression_logic(self, model_name, expected):
        regressor, regressor_logic = self._build_prediction_logic(
            model_name, neural_network_predictor.NeuralNetwork
        )

        traced_predictor = edsl.trace(regressor_logic)
        storage = {plc.name: {} for plc in regressor.host_placements}
        runtime = testing.LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in regressor.host_placements}

        input_x = np.array(
            [
                [
                    0.4595,
                    -0.8661,
                    1.7674,
                    1.9377,
                    0.3077,
                    -0.8155,
                    0.3508,
                    0.2848,
                    -1.8987,
                    0.3189,
                ]
            ],
            dtype=np.float64,
        )
        result_dict = runtime.evaluate_computation(
            computation=traced_predictor,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.asarray(expected)
        np.testing.assert_almost_equal(actual_result, expected_result, decimal=2)

    @parameterized.parameters(
        *zip(
            map(lambda x: x[0], _MODELS),
            itertools.repeat(neural_network_predictor.NeuralNetwork),
        ),
    )
    def test_serde(self, model_name, predictor_cls):
        regressor = self._build_NN_predictor(model_name, predictor_cls)
        predictor = regressor.predictor_factory()
        traced_predictor = edsl.trace(predictor)
        serialized = comp_utils.serialize_computation(traced_predictor)
        logical_comp_rustref = elk_compiler.compile_computation(serialized, [])
        logical_comp_rustbytes = logical_comp_rustref.to_bytes()
        pymoose.MooseComputation.from_bytes(logical_comp_rustbytes)
        # NOTE: could also dump to disk as follows (but we don't in the test)
        # logical_comp_rustref.to_disk(path)
        # pymoose.MooseComputation.from_disk(path)
