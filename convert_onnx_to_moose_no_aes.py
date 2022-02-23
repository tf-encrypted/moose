import argparse
import base64

import onnx

from pymoose import edsl
from pymoose import elk_compiler
from pymoose import predictors
from pymoose.computation import utils
from pymoose.predictors import predictor_utils
import numpy as np


def _convert_onnx_to_moose(compilation_step, onnx_proto, computation_path):
    if compilation_step == "logical":
        compilation_passes = []
    elif compilation_step == "physical":
        compilation_passes = [
            "typing",
            "full",
            "prune",
            "networking",
            "typing",
            "toposort",
        ]
    else:
        raise ValueError(
            "Compilation pass has to be `logical` or `pysical`, "
            f"found: {compilation_step}"
        )

    predictor = predictors.from_onnx(onnx_proto)

    @edsl.computation
    def predictor_no_aes():
        with predictor.alice:
            x = edsl.constant(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]] * 128))
            x_fixed = edsl.cast(x, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
        with predictor.replicated:
            y = predictor.forest_fn(x_fixed, predictor_utils.DEFAULT_FIXED_DTYPE)
            y = predictor.post_transform(y, predictor_utils.DEFAULT_FIXED_DTYPE)
            return predictor.handle_output(y, prediction_handler=predictor.bob)

    concrete_comp = edsl.trace(predictor_no_aes)
    comp_bin = utils.serialize_computation(concrete_comp)
    rust_compiled = elk_compiler.compile_computation(comp_bin, compilation_passes,)
    with open(computation_path, "w") as f:
        f.write(rust_compiled.to_textual())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ONNX to logical Moose computation"
    )
    parser.add_argument(
        "--compilation-step",
        type=str,
        default="logical",
        help="Select compilation step: logical or physical. Default: logical",
    )
    parser.add_argument("--onnx-path", type=str, help="Path to the onnx file.")
    parser.add_argument(
        "--moose-out", type=str, help="Where to save the logical Moose computation."
    )
    args = parser.parse_args()

    with open(args.onnx_path, "rb") as onnx_file:
        onnx_proto = onnx.load(onnx_file)

    _convert_onnx_to_moose(args.compilation_step, onnx_proto, args.moose_out)
