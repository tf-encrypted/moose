import argparse
import logging
import unittest

import numpy as np
import pytest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger

FIXED = pm.fixed(8, 27)
# Rust compiler currently supports only limited set of alternative precisions:
# FIXED = pm.fixed(14, 23)


def generate_data(seed, n_instances, n_features, coeff=3, shift=10):
    rng = np.random.default_rng()
    x_data = rng.normal(size=(n_instances, n_features))
    y_data = np.dot(x_data, np.ones(n_features) * coeff) + shift
    return x_data, y_data


def mse(y_pred, y_true):
    return pm.mean(pm.square(pm.sub(y_pred, y_true)), axis=0)


def mape(y_pred, y_true, y_true_inv):
    return pm.abs(pm.mul(pm.sub(y_pred, y_true), y_true_inv))


def ss_res(y_pred, y_true):
    squared_residuals = pm.square(pm.sub(y_true, y_pred))
    return pm.sum(squared_residuals, axis=0)


def ss_tot(y_true):
    y_mean = pm.mean(y_true)
    squared_deviations = pm.square(pm.sub(y_true, y_mean))
    return pm.sum(squared_deviations, axis=0)


def r_squared(ss_res, ss_tot):
    residuals_ratio = pm.div(ss_res, ss_tot)
    return pm.sub(pm.constant(1.0, dtype=pm.float64), residuals_ratio)


class LinearRegressionExample(parameterized.TestCase):
    def _build_linear_regression_example(self, metric_name="mse"):
        x_owner = pm.host_placement(name="x-owner")
        y_owner = pm.host_placement(name="y-owner")
        model_owner = pm.host_placement(name="model-owner")
        replicated_plc = pm.replicated_placement(
            players=[x_owner, y_owner, model_owner], name="replicated-plc"
        )

        @pm.computation
        def my_comp(
            x_uri: pm.Argument(placement=x_owner, vtype=pm.StringType()),
            y_uri: pm.Argument(placement=y_owner, vtype=pm.StringType()),
            w_uri: pm.Argument(placement=model_owner, vtype=pm.StringType()),
            metric_uri: pm.Argument(placement=model_owner, vtype=pm.StringType()),
            rsquared_uri: pm.Argument(placement=model_owner, vtype=pm.StringType()),
        ):
            with x_owner:
                X = pm.atleast_2d(
                    pm.load(x_uri, dtype=pm.float64), to_column_vector=True
                )
                # NOTE: what would be most natural to do is this:
                #     bias_shape = (slice(shape(X), begin=0, end=1), 1)
                #     bias = ones(bias_shape, dtype=float)
                # but this raises an issue about accomodating python native values in
                # the ASTTracer, something we've discussed and temporarily tabled in
                # the past. For now, we've decided to implement squeeze and unsqueeze
                # ops instead.
                # But we have a feeling this issue will continue to come up!
                bias_shape = pm.shape(X)[0:1]
                bias = pm.ones(bias_shape, dtype=pm.float64)
                reshaped_bias = pm.expand_dims(bias, 1)
                X_b = pm.concatenate([reshaped_bias, X], axis=1)
                A = pm.inverse(pm.dot(pm.transpose(X_b), X_b))
                B = pm.dot(A, pm.transpose(X_b))
                X_b = pm.cast(X_b, dtype=FIXED)
                B = pm.cast(B, dtype=FIXED)

            with y_owner:
                y_true = pm.atleast_2d(
                    pm.load(y_uri, dtype=pm.float64), to_column_vector=True
                )
                if metric_name == "mape":
                    y_true_inv = pm.cast(
                        pm.div(pm.constant(1.0, dtype=pm.float64), y_true),
                        dtype=FIXED,
                    )
                totals_ss = ss_tot(y_true)
                y_true = pm.cast(y_true, dtype=FIXED)

            with replicated_plc:
                w = pm.dot(B, y_true)
                y_pred = pm.dot(X_b, w)
                if metric_name == "mape":
                    metric_result = mape(y_pred, y_true, y_true_inv)
                else:
                    metric_result = mse(y_pred, y_true)
                residuals_ss = ss_res(y_pred, y_true)

            with model_owner:
                residuals_ss = pm.cast(residuals_ss, dtype=pm.float64)
                rsquared_result = r_squared(residuals_ss, totals_ss)

            with model_owner:
                w = pm.cast(w, dtype=pm.float64)
                metric_result = pm.cast(metric_result, dtype=pm.float64)
                res = (
                    pm.save(w_uri, w),
                    pm.save(metric_uri, metric_result),
                    pm.save(rsquared_uri, rsquared_result),
                )

            return res

        return my_comp, (x_owner, y_owner, model_owner, replicated_plc)

    def _linear_regression_eval(self, metric_name):
        linear_comp, _ = self._build_linear_regression_example(metric_name)

        x_data, y_data = generate_data(seed=42, n_instances=10, n_features=1)
        executors_storage = {
            "x-owner": {"x_data": x_data},
            "y-owner": {"y_data": y_data},
            "model-owner": {},
        }
        runtime = pm.LocalMooseRuntime(storage_mapping=executors_storage)
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
            runtime.read_value_from_storage("model-owner", "regression_weights"),
        )

    def test_linear_regression_mse(self):
        self._linear_regression_eval("mse")

    @pytest.mark.slow
    def test_linear_regression_mape(self):
        self._linear_regression_eval("mape")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
