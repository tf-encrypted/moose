import argparse
import base64

import onnx

from pymoose import edsl
from pymoose import elk_compiler
from pymoose import predictors
from pymoose.computation import utils


def _convert_onnx_to_moose(compilation_step, onnx_proto, computation_path, output_format):
    if compilation_step == "logical":
        compilation_passes = []
    elif compilation_step == "physical":
        compilation_passes = None
    else:
        raise ValueError(
            "Compilation pass has to be `logical` or `pysical`, "
            f"found: {compilation_step}"
        )

    pymoose_predictor = predictors.from_onnx(onnx_proto)
    aes_comp = pymoose_predictor.predictor_factory()
    concrete_comp = edsl.trace(aes_comp)
    comp_bin = utils.serialize_computation(concrete_comp)
    rust_compiled = elk_compiler.compile_computation(comp_bin, compilation_passes,)
    if output_format == "binary":
        with open(computation_path, "wb") as f:
            f.write(rust_compiled.to_bytes())
    elif output_format == "textual":
        with open(computation_path, "w") as f:
            f.write(rust_compiled.to_textual())
    else:
        raise(f"Unexpected output format: {output_format}, has to be `binary` or `textual`.")


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
    parser.add_argument("--output-format", type=str, default="textual", help="select output format binary or textual")
    parser.add_argument(
        "--moose-out", type=str, help="Where to save the logical Moose computation."
    )
    args = parser.parse_args()

    with open(args.onnx_path, "rb") as onnx_file:
        onnx_proto = onnx.load(onnx_file)

    _convert_onnx_to_moose(args.compilation_step, onnx_proto, args.moose_out, args.output_format)