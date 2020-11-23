import argparse
import asyncio
import logging
import os

from grpc.experimental import aio as grpc_aio

from moose.choreography.grpc import Choreography
from moose.executor.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.cape_broker import Networking

parser = argparse.ArgumentParser(description="Launch worker")
parser.add_argument("--port", type=int, default=50000)
parser.add_argument("--name", default=os.environ.get("CAPE_NAME", None))
parser.add_argument("--broker", default=os.environ.get("CAPE_BROKER", None))
parser.add_argument("--token", default=os.environ.get("CAPE_TOKEN", None))
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

logger = logging.getLogger("worker")
if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)
    logger.setLevel(level=logging.DEBUG)


if __name__ == "__main__":
    # Generate the keys()
    # pass them to the networking module
    networking = Networking(
        broker_host=args.broker, own_name=args.name, auth_token=args.token
    )
    executor = AsyncExecutor(networking=networking)

    grpc_aio.init_grpc_aio()
    grpc_server = grpc_aio.server()
    grpc_server.add_insecure_port(f"0.0.0.0:{args.port}")
    # Pass in the public key here
    choreography = Choreography(executor=executor, grpc_server=grpc_server)
    asyncio.get_event_loop().run_until_complete(grpc_server.start())
    logger.info("Worker started")

    asyncio.get_event_loop().run_until_complete(grpc_server.wait_for_termination())
    logger.info("Worker stopped")
