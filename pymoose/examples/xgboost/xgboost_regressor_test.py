import argparse
import json
import logging
import pathlib

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime

from .xgboost_regressor import XGBoostForestRegressor


class XGBoostReplicatedExample(parameterized.TestCase):
    def test_xgboost_regression_example_execute(self):
        input_x = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float64)

        with pathlib.Path(
            "./pymoose/examples/xgboost/xgboost_regression_2_trees.json"
        ) as p:
            with open(p) as f:
                forest_json = json.load(f)

        forest = XGBoostForestRegressor.from_json(forest_json)

        forest_predict = forest.predictor_factory(fixedpoint_dtype=edsl.fixed(8, 27))
        traced_model_comp = edsl.trace(forest_predict)

        storage = {plc.name: {} for plc in forest.host_placements}
        runtime = LocalMooseRuntime(storage_mapping=storage)
        role_assignment = {plc.name: plc.name for plc in forest.host_placements}
        result_dict = runtime.evaluate_computation(
            computation=traced_model_comp,
            role_assignment=role_assignment,
            arguments={"x": input_x},
        )
        actual_result = list(result_dict.values())[0]
        expected_result = np.array([26.6495275, 88.423399], dtype=np.float64)
        np.testing.assert_almost_equal(actual_result, expected_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
