import argparse
import ast
import json
import logging

from logger import get_logger
from logger import set_logger

get_logger().setLevel(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Local computation")
parser.add_argument("--input-file", type=str, default="None")
parser.add_argument("--output-file", type=str, default=".")
parser.add_argument("--session-id", type=str, default=0)
parser.add_argument("--device", type=str, default="inputter0")

args = parser.parse_args()


def foo(x=0, y=0):
    return x + y + 1


if __name__ == "__main__":
    if args.input_file == "None":
        output = foo()
    else:
        with open(args.input_file, "r") as f:
            inputs = json.loads(f.read())
        output = foo(*inputs["inputs"])

    # [TODO] find a more agnostic way to serialize output
    output_dict = {args.session_id: output}
    with open(args.output_file, "w") as f:
        f.write(json.dumps(output_dict))

    get_logger().debug(f"Computation completed on device {args.device}")
