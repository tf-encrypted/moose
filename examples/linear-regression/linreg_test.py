import argparse
import logging
import unittest

import numpy as np

from moose.edsl import Argument
from moose.edsl import computation
from moose.edsl import concatenate
from moose.edsl import constant
from moose.edsl import div
from moose.edsl import dot
from moose.edsl import expand_dims
from moose.edsl import host_placement
from moose.edsl import inverse
from moose.edsl import load
from moose.edsl import mean
from moose.edsl import ones
from moose.edsl import save
from moose.edsl import shape
from moose.edsl import slice
from moose.edsl import square
from moose.edsl import sub
from moose.edsl import sum
from moose.edsl import trace
from moose.edsl import transpose
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.memory import Networking
from moose.runtime import TestRuntime as Runtime


def generate_data(seed, n_instances, n_features, n_targets, coeff=3, shift=10):
    rng = np.random.default_rng()
    x_data = rng.normal(size=(n_instances, n_features))
    y_data = x_data * coeff + shift
    return x_data, y_data


def mse(y_pred, y_true):
    return mean(square(sub(y_pred, y_true)), axis=0)


def r_squared(y_pred, y_true):
    y_mean = mean(y_true)
    ss_tot = sum(square(sub(y_true, y_mean)), axis=0)
    ss_res = sum(square(sub(y_true, y_pred)), axis=0)
    # NOTE this division is going to be a problem in replicated dialect, instead
    # we could reveal ss_res and ss_tot to the model owner then do the division
    return sub(constant(1.0), div(ss_res, ss_tot))


class LinearRegressionExample(unittest.TestCase):
    def test_linear_regression_example(self):
        x_owner = host_placement(name="x-owner")
        y_owner = host_placement(name="y-owner")
        model_owner = host_placement(name="model-owner")
        trusted_computer = host_placement(name="trusted-computer")

        @computation
        def my_comp(
            x_uri: Argument(placement=x_owner, datatype=str),
            y_uri: Argument(placement=y_owner, datatype=str),
            w_uri: Argument(placement=model_owner, datatype=str),
            mse_uri: Argument(placement=model_owner, datatype=str),
            rsquared_uri: Argument(placement=model_owner, datatype=str),
        ):

            with x_owner:
                X = load(x_uri, dtype=float)
                # NOTE: what would be most natural to do is this:
                #     bias_shape = (slice(shape(X), begin=0, end=1), 1)
                #     bias = ones(bias_shape, dtype=float)
                # but this raises an issue about accomodating python native values in
                # the ASTTracer, something we've discussed and temporarily tabled in
                # the past. For now, we've decided to implement squeeze and unsqueeze
                # ops instead.
                # But we have a feeling this issue will continue to come up!
                bias = ones(slice(shape(X), begin=0, end=1), dtype=float)
                reshaped_bias = expand_dims(bias, 1)
                X_b = concatenate([reshaped_bias, X], axis=1)
                A = inverse(dot(transpose(X_b), X_b))
                B = dot(A, transpose(X_b))

            with y_owner:
                y_true = load(y_uri, dtype=float)

            with trusted_computer:
                w = dot(B, y_true)
                y_pred = dot(X_b, w)
                mse_result = mse(y_pred, y_true)

            with model_owner:
                # NOTE: we can alternatively compute the SS terms on trusted_computer,
                # and only do the division & subtraction here
                rsquared_result = r_squared(y_pred, y_true)

            with model_owner:
                res = (
                    save(w, w_uri),
                    save(mse_result, mse_uri),
                    save(rsquared_result, rsquared_uri),
                )

            return res

        concrete_comp = trace(my_comp)

        x_data, y_data = generate_data(
            seed=42, n_instances=10, n_features=1, n_targets=1
        )
        networking = Networking()
        x_owner_executor = AsyncExecutor(networking, store={"x_data": x_data})
        y_owner_executor = AsyncExecutor(networking, store={"y_data": y_data})
        runtime = Runtime(
            networking=networking,
            backing_executors={
                x_owner.name: x_owner_executor,
                y_owner.name: y_owner_executor,
            },
        )
        runtime.evaluate_computation(
            concrete_comp,
            placement_instantiation={
                plc: plc.name
                for plc in [x_owner, y_owner, model_owner, trusted_computer]
            },
            arguments={
                "x_uri": "x_data",
                "y_uri": "y_data",
                "w_uri": "regression_weights",
                "mse_uri": "mse_result",
                "rsquared_uri": "rsquared_result",
            },
        )

        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

        from opentelemetry.exporter.jaeger import JaegerSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchExportSpanProcessor
        from opentelemetry.trace import set_tracer_provider

        trace_provider = TracerProvider()
        trace_provider.add_span_processor(
            BatchExportSpanProcessor(
                JaegerSpanExporter(
                    service_name="moose", agent_host_name="localhost", agent_port=6831,
                )
            )
        )
        set_tracer_provider(trace_provider)

    unittest.main()
