import argparse
import logging
import unittest

from moose.edsl import Argument
from moose.edsl import computation
from moose.edsl import concatenate
from moose.edsl import constant
from moose.edsl import div
from moose.edsl import dot
from moose.edsl import host_placement
from moose.edsl import inverse
from moose.edsl import load
from moose.edsl import mean
from moose.edsl import nth
from moose.edsl import ones
from moose.edsl import save
from moose.edsl import shape
from moose.edsl import slice
from moose.edsl import square
from moose.edsl import sub
from moose.edsl import sum
from moose.edsl import trace
from moose.edsl import transpose
from moose.logger import get_logger
from moose.runtime import TestRuntime as Runtime


def mse(y_pred, y_true):
    return div(sum(square(sub(y_pred, y_true)), axis=1), nth(shape(y_pred), index=0))


def r_squared(y_pred, y_true):
    y_mean = mean(y_true)
    ss_tot = sum(square(sub(y_true, y_mean)), axis=1)
    ss_res = sum(square(sub(y_true, y_pred)), axis=1)
    # NOTE this division is going to be a problem
    # instead we could reveal ss_res and ss_tot to the
    # model owner then do the division
    return sub(constant(1, dtype=float), div(ss_res, ss_tot))


class LinearRegressionExample(unittest.TestCase):
    # @unittest.skip
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
                X = load(x_uri, dtype=float)  # , x_source.selected_columns)
                bias = ones(slice(shape(X), begin=0, end=1), dtype=float)
                X_b = concatenate([bias, X])
                A = inverse(dot(transpose(X_b), X_b))
                B = dot(A, transpose(X_b))

            with y_owner:
                y_true = load(y_uri, dtype=float)  # , y_source.selected_columns)

            with model_owner:
                w = dot(B, y_true)

            with trusted_computer:
                y_pred = dot(X_b, w)
                mse_result = mse(y_pred, y_true)
                # rsquared_result = r_squared(y_pred, y_true)

            with model_owner:
                rsquared_result = r_squared(y_pred, y_true)

            with model_owner:
                res = (
                    save(w, w_uri),
                    save(mse_result, mse_uri),
                    save(rsquared_result, rsquared_uri),
                )

            return res

        concrete_comp = trace(my_comp)
        runtime = Runtime()
        runtime.evaluate_computation(
            concrete_comp,
            placement_instantiation={
                plc: plc.name
                for plc in [x_owner, y_owner, model_owner, trusted_computer]
            },
            arguments={
                "x_uri": "",
                "y_uri": "",
                "w_uri": "",
                "mse_uri": "",
                "r_squared": "",
            },
        )

        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
