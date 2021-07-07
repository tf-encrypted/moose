import argparse
import logging
import unittest

import numpy as np
import pytest
from absl.testing import parameterized

from moose import edsl
from moose.computation import utils
from moose.computation.standard import StringType
from moose.logger import get_logger
from moose.testing import LocalMooseRuntime

FIXED = edsl.fixed(8, 27)


def generate_data(seed, n_instances, n_features, coeff=3, shift=10):
    rng = np.random.default_rng()
    x_data = rng.normal(size=(n_instances, n_features))
    y_data = np.dot(x_data, np.ones(n_features) * coeff) + shift
    return x_data, y_data


def mse(y_pred, y_true):
    return edsl.mean(edsl.square(edsl.sub(y_pred, y_true)), axis=0)


def mape(y_pred, y_true, y_true_inv):
    return edsl.abs(edsl.mul(edsl.sub(y_pred, y_true), y_true_inv))


def ss_res(y_pred, y_true):
    squared_residuals = edsl.square(edsl.sub(y_true, y_pred))
    return edsl.sum(squared_residuals, axis=0)


def ss_tot(y_true):
    y_mean = edsl.mean(y_true)
    squared_deviations = edsl.square(edsl.sub(y_true, y_mean))
    return edsl.sum(squared_deviations, axis=0)


def r_squared(ss_res, ss_tot):
    residuals_ratio = edsl.div(ss_res, ss_tot)
    return edsl.sub(edsl.constant(1.0, dtype=edsl.float64), residuals_ratio)


class LinearRegressionExample(parameterized.TestCase):
    def _build_linear_regression_example(self, metric_name="mse", compiler_passes=None):
        x_owner = edsl.host_placement(name="x-owner")
        y_owner = edsl.host_placement(name="y-owner")
        model_owner = edsl.host_placement(name="model-owner")
        replicated_plc = edsl.replicated_placement(
            players=[x_owner, y_owner, model_owner], name="replicated-plc"
        )

        @edsl.computation
        def my_comp(
            x_uri: edsl.Argument(placement=x_owner, vtype=StringType()),
            y_uri: edsl.Argument(placement=y_owner, vtype=StringType()),
            w_uri: edsl.Argument(placement=model_owner, vtype=StringType()),
            metric_uri: edsl.Argument(placement=model_owner, vtype=StringType()),
            rsquared_uri: edsl.Argument(placement=model_owner, vtype=StringType()),
        ):
            with x_owner:
                X = edsl.atleast_2d(
                    edsl.load(x_uri, dtype=edsl.float64), to_column_vector=True
                )
                # NOTE: what would be most natural to do is this:
                #     bias_shape = (slice(shape(X), begin=0, end=1), 1)
                #     bias = ones(bias_shape, dtype=float)
                # but this raises an issue about accomodating python native values in
                # the ASTTracer, something we've discussed and temporarily tabled in
                # the past. For now, we've decided to implement squeeze and unsqueeze
                # ops instead.
                # But we have a feeling this issue will continue to come up!
                bias_shape = edsl.slice(edsl.shape(X), begin=0, end=1)
                bias = edsl.ones(bias_shape, dtype=edsl.float64)
                reshaped_bias = edsl.expand_dims(bias, 1)
                X_b = edsl.concatenate([reshaped_bias, X], axis=1)
                A = edsl.inverse(edsl.dot(edsl.transpose(X_b), X_b))
                B = edsl.dot(A, edsl.transpose(X_b))
                X_b = edsl.cast(X_b, dtype=FIXED)
                B = edsl.cast(B, dtype=FIXED)

            with y_owner:
                y_true = edsl.atleast_2d(
                    edsl.load(y_uri, dtype=edsl.float64), to_column_vector=True
                )
                if metric_name == "mape":
                    y_true_inv = edsl.cast(
                        edsl.div(edsl.constant(1.0, dtype=edsl.float64), y_true),
                        dtype=FIXED,
                    )
                totals_ss = ss_tot(y_true)
                y_true = edsl.cast(y_true, dtype=FIXED)

            with replicated_plc:
                w = edsl.dot(B, y_true)
                y_pred = edsl.dot(X_b, w)
                if metric_name == "mape":
                    metric_result = mape(y_pred, y_true, y_true_inv)
                else:
                    metric_result = mse(y_pred, y_true)
                residuals_ss = ss_res(y_pred, y_true)

            with model_owner:
                residuals_ss = edsl.cast(residuals_ss, dtype=edsl.float64)
                rsquared_result = r_squared(residuals_ss, totals_ss)

            with model_owner:
                w = edsl.cast(w, dtype=edsl.float64)
                metric_result = edsl.cast(metric_result, dtype=edsl.float64)
                res = (
                    edsl.save(w_uri, w),
                    edsl.save(metric_uri, metric_result),
                    edsl.save(rsquared_uri, rsquared_result),
                )

            return res

        return my_comp, (x_owner, y_owner, model_owner, replicated_plc)

    def _linear_regression_eval(self, metric_name):
        linear_comp, placements = self._build_linear_regression_example(metric_name)

        x_data, y_data = generate_data(seed=42, n_instances=10, n_features=1)
        executors_storage = {
            "x-owner": {"x_data": x_data},
            "y-owner": {"y_data": y_data},
            "model-owner": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=executors_storage)
        _ = runtime.evaluate_computation(
            computation=linear_comp,
            role_assignment={
                "x-owner": "x-owner",
                "y-owner": "y-owner",
                "model-owner": "model-owner",
            },
            arguments={
                "x_uri": "x_data",
                "y_uri": "y_data",
                "w_uri": "regression_weights",
                "metric_uri": "metric_result",
                "rsquared_uri": "rsquared_result",
            },
        )
        print(
            "Done: \n",
            runtime.get_value_from_storage("model-owner", "regression_weights"),
        )

    def test_linear_regression_mse(self):
        self._linear_regression_eval("mse")

    @pytest.mark.slow
    def test_linear_regression_mape(self):
        self._linear_regression_eval("mape")

    @parameterized.parameters(True, False)
    def test_linear_regression_serde(self, compiled):
        passes_arg = None if compiled else []
        comp, _ = self._build_linear_regression_example(compiler_passes=passes_arg)
        compiled_comp = edsl.trace_and_compile(comp)
        serialized = utils.serialize_computation(compiled_comp)
        deserialized = utils.deserialize_computation(serialized)
        assert compiled_comp == deserialized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
