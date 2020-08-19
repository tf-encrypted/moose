import argparse
import json
import logging

from logger import get_logger
from logger import set_logger

get_logger().setLevel(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Local computation")
parser.add_argument("--inputs", type=int, default=0)
parser.add_argument("--session-id", type=str, default=0)
parser.add_argument("--device", type=str, default="inputter0")

args = parser.parse_args()


def foo(x):
    return x + 1


if __name__ == "__main__":
    output = foo(args.inputs)

    get_logger().debug(f"Computation completed on device {args.device}")

    # [TODO] find a more agnostic way to serialize output
    output_store = {args.session_id: output}
    # [TODO] check if file already exist. if yes, store new values in this file
    outputfile = "/tmp/" + args.device + "_" + "data_output.json"
    with open(outputfile, "w") as fp:
        json.dump(output_store, fp)
