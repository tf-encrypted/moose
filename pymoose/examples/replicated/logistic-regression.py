import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from onnxmltools import convert_sklearn
from skl2onnx.common import data_types as onnx_dtypes

from pymoose import edsl
from pymoose import testing
from pymoose.predictors import linear_predictor
from pymoose.predictors import predictor_utils


def _build_prediction_logic(onnx_proto):
    predictor = linear_predictor.LinearClassifier.from_onnx(onnx_proto)

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



if __name__ == '__main__':
    random_state = 99
    n_sample_train = 10000
    n_sample_test = 100
    n_features = 10
    n_informative = 10
    n_redundant = 0
    n_classes = 2

    X_train, y_train = make_classification(n_samples=100, n_features=n_features, n_informative=n_informative, n_classes=n_classes, n_redundant=n_redundant, random_state=random_state)
    X_test, y_test = make_classification(n_samples=100, n_features=n_features, n_informative=n_informative, n_classes=n_classes, n_redundant=n_redundant, random_state=random_state)
    lg = LogisticRegression()
    trained_model = lg.fit(X_train, y_train)

    initial_type = ("float_input", onnx_dtypes.FloatTensorType([None, n_features]))
    onnx_proto = convert_sklearn(trained_model, initial_types=[initial_type])

    regressor, regressor_logic = _build_prediction_logic(onnx_proto)

    traced_predictor = edsl.trace(regressor_logic)
    storage = {plc.name: {} for plc in regressor.host_placements}
    runtime = testing.LocalMooseRuntime(storage_mapping=storage)
    role_assignment = {plc.name: plc.name for plc in regressor.host_placements}

    result_dict = runtime.evaluate_computation(
        computation=traced_predictor,
        role_assignment=role_assignment,
        arguments={"x": X_test},
    )
    actual_result = list(result_dict.values())[0]
    expected = trained_model.predict_proba(X_test)
    expected_result = np.array(expected)
    np.testing.assert_almost_equal(actual_result, expected_result, decimal=4)

