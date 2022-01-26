import argparse
import itertools
import json
import logging
import pathlib
import time
from signal import valid_signals

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from onnx import load_model

import pymoose
from pymoose import edsl
from pymoose import elk_compiler
from pymoose import predictors
from pymoose.computation import utils as comp_utils
from pymoose.logger import get_logger
from pymoose.predictors import linear_predictor
from pymoose.predictors import predictor_utils
from pymoose.predictors import tree_ensemble
from pymoose.testing import LocalMooseRuntime


def _build_forest_from_onnx(model_name, predictor_cls):
    root_path = pathlib.Path(__file__).parent.absolute()
    fixture_path = root_path / "fixtures" / f"{model_name}.onnx"
    with open(fixture_path, "rb") as model_fixture:
        forest_onnx = load_model(model_fixture)
    forest_model = predictor_cls.from_onnx(forest_onnx)
    return forest_model


def _build_forest_from_json(predictor_cls):
    root_path = pathlib.Path(__file__).parent.absolute()
    with root_path / "fixtures" / "xgboost_regressor.json" as p:
        with open(p) as f:
            forest_json = json.load(f)
    forest_model = predictor_cls.from_json(forest_json)
    return forest_model


def _build_prediction_logic(model_name, onnx_or_json, predictor_cls):
    if onnx_or_json == "onnx":
        predictor = _build_forest_from_onnx(model_name, predictor_cls)
    elif onnx_or_json == "json":
        predictor = _build_forest_from_json()
    else:
        raise ValueError()

    @edsl.computation
    def predictor_no_aes(x: edsl.Argument(predictor.alice, dtype=edsl.float64)):
        with predictor.alice:
            x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
        with predictor.replicated:
            y = predictor.forest_fn(x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE)
            y = predictor.post_transform(y, predictor_utils.DEFAULT_FIXED_DTYPE)
        return predictor.handle_output(y, prediction_handler=predictor.bob)

    return predictor, predictor_no_aes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor Benchmark")
    parser.add_argument(
        "--onnx-model", type=str, required=True, help="ONNX file name",
    )
    parser.add_argument("--model-type", type=str, required=True)
    args = parser.parse_args()

    print("ONNX Model: ", args.onnx_model)

    input_x = np.array([[0, 1, 1, 0]], dtype=np.float64)

    # TODO simplify with predictor.from_onnx
    if args.model_type == "linear_regressor":
        predictor_cls = linear_predictor.LinearRegressor
    elif args.model_type == "linear_classifier":
        predictor_cls = linear_predictor.LinearClassifier
    elif args.model_type == "tree_ensemble_regressor":
        predictor_cls = tree_ensemble.TreeEnsembleRegressor
    elif args.model_type == "tree_ensemble_classifier":
        predictor_cls = tree_ensemble.TreeEnsembleClassifier
    else:
        raise ValueError(f"Got unexpected model type: {args.model_type}")

    predictor, predictor_logic = _build_prediction_logic(
        args.onnx_model, "onnx", predictor_cls
    )
    traced_model_comp = edsl.trace(predictor_logic)
    storage = {plc.name: {} for plc in predictor.host_placements}
    runtime = LocalMooseRuntime(storage_mapping=storage)
    role_assignment = {plc.name: plc.name for plc in predictor.host_placements}

    compiler_passes = ["typing", "full", "prune", "networking", "toposort"]

    comp_bin = comp_utils.serialize_computation(traced_model_comp)

    start_compile = time.perf_counter()
    rust_compiled = elk_compiler.compile_computation(
        comp_bin, ["typing", "full", "prune", "networking", "typing", "toposort"]
    )
    end_compile = time.perf_counter()
    print(f"Compile time {end_compile - start_compile:0.4f} seconds")

    start_evaluate = time.perf_counter()
    result_dict = runtime.evaluate_compiled(
        comp_bin=rust_compiled,
        role_assignment=role_assignment,
        arguments={"x": input_x},
    )
    end_evaluate = time.perf_counter()
    print(f"Evaluate time {end_evaluate - start_evaluate:0.4f} seconds")

    actual_result = list(result_dict.values())[0]
