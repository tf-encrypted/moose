import argparse
import asyncio
import logging
import os

from moose.logger import get_logger
from moose.networking.grpc_broker import Broker

parser = argparse.ArgumentParser(description="Launch broker")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=str, default="40000")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

if __name__ == "__main__":

    broker = Broker(host=args.host, port=args.port, allow_insecure_networking=True,)

    asyncio.get_event_loop().run_until_complete(board.start())
    get_logger().info("Started")

    asyncio.get_event_loop().run_until_complete(board.wait())
    get_logger().info("Stopped")
