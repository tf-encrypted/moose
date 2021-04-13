import argparse
import logging
import unittest

import numpy as np

from moose import edsl
from moose.computation.standard import StringType
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.memory import Networking
from moose.storage.memory import MemoryDataStore
from moose.testing import TestRuntime as Runtime


def generate_data(seed, n_instances, n_features, coeff=3, shift=10):
    rng = np.random.default_rng()
    x_data = rng.normal(size=(n_instances, n_features))
    y_data = np.dot(x_data, np.ones(n_features) * coeff) + shift
    return x_data, y_data


def mse(y_pred, y_true):
    return edsl.mean(edsl.square(edsl.sub(y_pred, y_true)), axis=0)


def ss_res(y_pred, y_true):
    squared_residuals = edsl.square(edsl.sub(y_true, y_pred))
    return edsl.sum(squared_residuals, axis=0)


def ss_tot(y_true):
    y_mean = edsl.mean(y_true)
    squared_deviations = edsl.square(edsl.sub(y_true, y_mean))
    return edsl.sum(squared_deviations, axis=0)


def r_squared(ss_res, ss_tot):
    residuals_ratio = edsl.div(ss_res, ss_tot)
    return edsl.sub(edsl.constant(1.0, dtype=edsl.float32), residuals_ratio)


class LinearRegressionExample(unittest.TestCase):
    def test_linear_regression_example(self):
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
            mse_uri: edsl.Argument(placement=model_owner, vtype=StringType()),
            rsquared_uri: edsl.Argument(placement=model_owner, vtype=StringType()),
        ):

            with x_owner:
                X = edsl.atleast_2d(
                    edsl.load(x_uri, dtype=edsl.float32), to_column_vector=True
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
                bias = edsl.ones(bias_shape, dtype=edsl.float32)
                reshaped_bias = edsl.expand_dims(bias, 1)
                X_b = edsl.concatenate([reshaped_bias, X], axis=1)
                A = edsl.inverse(edsl.dot(edsl.transpose(X_b), X_b))
                B = edsl.dot(A, edsl.transpose(X_b))

            with y_owner:
                y_true = edsl.atleast_2d(
                    edsl.load(y_uri, dtype=edsl.float32), to_column_vector=True
                )
                totals_ss = ss_tot(y_true)

            with replicated_plc:
                w = edsl.dot(B, y_true)
                y_pred = edsl.dot(X_b, w)
                mse_result = mse(y_pred, y_true)
                residuals_ss = ss_res(y_pred, y_true)

            with model_owner:
                rsquared_result = r_squared(residuals_ss, totals_ss)

            with model_owner:
                res = (
                    edsl.save(w_uri, w),
                    edsl.save(mse_uri, mse_result),
                    edsl.save(rsquared_uri, rsquared_result),
                )

            return res

        concrete_comp = edsl.trace(my_comp)

        x_data, y_data = generate_data(seed=42, n_instances=10, n_features=1)
        networking = Networking()
        x_owner_storage = MemoryDataStore({"x_data": x_data})
        x_owner_executor = AsyncExecutor(networking, storage=x_owner_storage)
        y_owner_storage = MemoryDataStore({"y_data": y_data})
        y_owner_executor = AsyncExecutor(networking, storage=y_owner_storage)
        model_owner_storage = MemoryDataStore()
        model_owner_executor = AsyncExecutor(networking, storage=model_owner_storage)
        runtime = Runtime(
            networking=networking,
            backing_executors={
                x_owner.name: x_owner_executor,
                y_owner.name: y_owner_executor,
                model_owner.name: model_owner_executor,
            },
        )
        runtime.evaluate_computation(
            concrete_comp,
            placement_instantiation={
                plc: plc.name for plc in [x_owner, y_owner, model_owner]
            },
            arguments={
                "x_uri": "x_data",
                "y_uri": "y_data",
                "w_uri": "regression_weights",
                "mse_uri": "mse_result",
                "rsquared_uri": "rsquared_result",
            },
        )

        print("Done: \n", model_owner_storage.store["regression_weights"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
