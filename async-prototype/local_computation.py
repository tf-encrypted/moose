import argparse
import json
import logging

from logger import get_logger
from logger import set_logger

get_logger().setLevel(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Local computation")
parser.add_argument("--session-id", type=str, default=0)
parser.add_argument("--device", type=str, default="inputter0")

args = parser.parse_args()


def foo():
    x = 1
    y = 2
    z = x + y
    return z


if __name__ == "__main__":
    output = foo()

    get_logger().debug(f"Computation complete on device {args.device}")

    data_store = {args.session_id: output}

    filename = "/tmp/" + "_" + args.device + "data_store.json"
    with open(filename, "w") as fp:
        json.dump(data_store, fp)
