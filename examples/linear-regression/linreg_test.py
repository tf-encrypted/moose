import argparse
import logging
import unittest

import numpy as np

import moose as moo
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
    return moo.mean(moo.square(moo.sub(y_pred, y_true)), axis=0)


def ss_res(y_pred, y_true):
    squared_residuals = moo.square(moo.sub(y_true, y_pred))
    return moo.sum(squared_residuals, axis=0)


def ss_tot(y_true):
    y_mean = moo.mean(y_true)
    squared_deviations = moo.square(moo.sub(y_true, y_mean))
    return moo.sum(squared_deviations, axis=0)


def r_squared(ss_res, ss_tot):
    residuals_ratio = moo.div(ss_res, ss_tot)
    return moo.sub(moo.constant(1.0), residuals_ratio)


class LinearRegressionExample(unittest.TestCase):
    def test_linear_regression_example(self):
        x_owner = moo.host_placement(name="x-owner")
        y_owner = moo.host_placement(name="y-owner")
        model_owner = moo.host_placement(name="model-owner")
        replicated_plc = moo.replicated_placement(
            players=[x_owner, y_owner, model_owner], name="replicated-plc"
        )

        @moo.computation
        def my_comp(
            x_uri: moo.Argument(placement=x_owner, dtype=str),
            y_uri: moo.Argument(placement=y_owner, dtype=str),
            w_uri: moo.Argument(placement=model_owner, dtype=str),
            mse_uri: moo.Argument(placement=model_owner, dtype=str),
            rsquared_uri: moo.Argument(placement=model_owner, dtype=str),
        ):

            with x_owner:
                X = moo.atleast_2d(
                    moo.load(x_uri, dtype=float), to_column_vector=True
                )
                # NOTE: what would be most natural to do is this:
                #     bias_shape = (slice(shape(X), begin=0, end=1), 1)
                #     bias = ones(bias_shape, dtype=float)
                # but this raises an issue about accomodating python native values in
                # the ASTTracer, something we've discussed and temporarily tabled in
                # the past. For now, we've decided to implement squeeze and unsqueeze
                # ops instead.
                # But we have a feeling this issue will continue to come up!
                bias = moo.ones(moo.slice(moo.shape(X), begin=0, end=1), dtype=float)
                reshaped_bias = moo.expand_dims(bias, 1)
                X_b = moo.concatenate([reshaped_bias, X], axis=1)
                A = moo.inverse(moo.dot(moo.transpose(X_b), X_b))
                B = moo.dot(A, moo.transpose(X_b))

            with y_owner:
                y_true = moo.atleast_2d(
                    moo.load(y_uri, dtype=float), to_column_vector=True
                )
                totals_ss = ss_tot(y_true)

            with replicated_plc:
                w = moo.dot(B, y_true)
                y_pred = moo.dot(X_b, w)
                mse_result = mse(y_pred, y_true)
                residuals_ss = ss_res(y_pred, y_true)

            with model_owner:
                rsquared_result = r_squared(residuals_ss, totals_ss)

            with model_owner:
                res = (
                    moo.save(w_uri, w),
                    moo.save(mse_uri, mse_result),
                    moo.save(rsquared_uri, rsquared_result),
                )

            return res

        concrete_comp = moo.trace(my_comp)

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
