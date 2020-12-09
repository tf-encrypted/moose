import argparse
import asyncio
import logging
import os

import pysodium

from moose.choreography.cape_coordinator import Choreography
from moose.executor import AsyncExecutor
from moose.logger import get_logger
from moose.networking.cape_broker import Networking

parser = argparse.ArgumentParser(description="Launch worker")
parser.add_argument("--port", type=int, default=50000)
parser.add_argument(
    "--broker",
    default=os.environ.get("CAPE_BROKER", "https://demo.capeprivacy.com/broker"),
)
parser.add_argument(
    "--coordinator",
    default=os.environ.get("CAPE_COORDINATOR", "https://demo.capeprivacy.com"),
)
parser.add_argument("--token", default=os.environ.get("CAPE_TOKEN", None))
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

logger = logging.getLogger("worker")
if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)
    logger.setLevel(level=logging.DEBUG)

if __name__ == "__main__":
    public_key, secret_key = pysodium.crypto_box_keypair()

    networking = Networking(
        broker_host=args.broker,
        coordinator_host=args.coordinator,
        auth_token=args.token,
        public_key=public_key,
        secret_key=secret_key,
    )
    executor = AsyncExecutor(networking=networking)
    cape_choreography = Choreography(
        executor=executor, auth_token=args.token, public_key=public_key
    )

    asyncio.get_event_loop().run_until_complete(cape_choreography.run())
