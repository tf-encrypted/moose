'''
This script benchmarks accuracy of linear regression against Sklearn to test if loss of precision occurs.
Local computations, replicated, no aes encryption
'''

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from onnxmltools import convert_sklearn
from skl2onnx.common import data_types as onnx_dtypes

from pymoose import edsl
from pymoose import testing
from pymoose.predictors import linear_predictor
from pymoose.predictors import predictor_utils


def _build_prediction_logic(onnx_proto):
    predictor = linear_predictor.LinearRegressor.from_onnx(onnx_proto)

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
    
def benchmark(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    trained_model = lr.fit(X_train, y_train)

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
    actual_result = actual_result.reshape((n_sample_test, ))
    expected = trained_model.predict(X_test)
    expected_result = np.array(expected)

    min_X = np.min(X_train)
    max_X = np.max(X_train)

    match_2_decimals = np.isclose(actual_result, expected_result, atol=1e-2).all() # Do outputs match up to 2 decimal points
    match_4_decimals = np.isclose(actual_result, expected_result, atol=1e-4).all() # Do outputs match up to 4 decimal points

    mean_abs_diff = np.mean(np.abs(actual_result - expected_result)) # Mean absolute difference
    max_abs_diff = np.max(np.abs(actual_result - expected_result)) # Max absolute difference
    std_abs_diff = np.std(np.abs(actual_result - expected_result)) # Standard deviation of absolute difference

    mean_rel_diff = np.max(np.abs((actual_result - expected_result)/expected_result)) # Mean relative difference
    max_rel_diff = np.mean(np.abs((actual_result - expected_result)/expected_result)) # Max relative difference
    std_rel_diff = np.std(np.abs((actual_result - expected_result)/expected_result)) # Mean relative difference

    # record results
    results.loc[results.shape[0]] = [min_X, max_X, n_features, n_informative, n_redundant, noise, match_2_decimals, match_4_decimals, mean_abs_diff, max_abs_diff,
    std_abs_diff, mean_rel_diff, max_rel_diff, std_rel_diff]

if __name__ == '__main__':
    # baseline default benchmark settings
    random_state = 99
    n_sample_train = 1000
    n_sample_test = 100
    n_features = 10
    n_informative = 10
    n_redundant = 0
    noise = 0.0

    # varying inputs and parameters
    noise_level = [0.0, 1.0, 1.5, 10.0, 100.0]
    scales = [0.00001, 0.0001, 0.001, 0.1, 10.0, 100.0, 1000.0, 100000.0, 1000000.0]

    # create dataset
    X_train, y_train = make_regression(n_samples=n_sample_train, n_features=n_features, n_informative=n_informative, noise=noise, random_state=random_state)
    X_test, y_test = make_regression(n_samples=n_sample_test, n_features=n_features, n_informative=n_informative, noise=noise, random_state=random_state)

    # table to store accuracy analysis results
    results = pd.DataFrame(columns=["min_X", "max_X", "features", "informative_features", "redundant_features", "noise", "match_2_decimals", "match_4_decimals", "mean_abs_diff", "max_abs_diff",
    "std_abs_diff", "mean_rel_diff", "max_rel_diff", "std_rel_diff"])
    
    # test robustness to varying noise in dataset
    for noise in noise_level:
        benchmark(X_train, y_train, X_test, y_test)

    # test robustness to varying features magnitude in dataset
    for scale in scales:
        benchmark(X_train * scale, y_train, X_test * scale, y_test)

    # save results to csv
    results.to_csv("linear_regression_accuracy_analysis.csv")
    print(results)