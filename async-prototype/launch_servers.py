import argparse
import asyncio
import logging

from grpc.experimental import aio

from logger import get_logger
from logger import set_logger
from channels import Server
from channels import ChannelManager
from executor import AsyncKernelBasedExecutor


parser = argparse.ArgumentParser(description="Launch servers")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=str, default="50000")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

if __name__ == "__main__":
    aio.init_grpc_aio()

    cluster_spec = {
        "inputter0": "localhost:50051",
        "inputter1": "localhost:50052",
        "aggregator": "localhost:50053",
        "outputter": "localhost:50054",
    }

    channel_manager = ChannelManager(cluster_spec)

    executor = AsyncKernelBasedExecutor(
        name="remote",
        store={args.valuename: args.value},
        channel_manager=channel_manager,
    )

    loop = asyncio.get_event_loop()

    get_logger().info(f"Starting on {args.host}:{args.port}")
    server = Server(args.host, args.port)

    loop.run_until_complete(server.start())
    get_logger().info("Started")
    loop.run_until_complete(server.wait())
    get_logger().info("Stopped")
