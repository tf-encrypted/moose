import argparse
import asyncio
import logging

from moose.logger import get_logger
from moose.worker import Worker

parser = argparse.ArgumentParser(description="Launch worker")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=50000)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

if __name__ == "__main__":
    worker = Worker(
        host=args.host, port=args.port, allow_insecure_networking=True,  # TODO
    )

    asyncio.get_event_loop().run_until_complete(worker.start())
    get_logger().info("Started")

    asyncio.get_event_loop().run_until_complete(worker.wait_for_termination())
    get_logger().info("Stopped")
