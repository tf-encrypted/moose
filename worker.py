import argparse
import asyncio
import logging

from grpc.experimental import aio

from cluster.cluster_spec import load_cluster_spec
from moose.channels.grpc import ChannelManager
from moose.executor.executor import KernelBasedExecutor
from moose.logger import get_logger
from moose.server import Server

parser = argparse.ArgumentParser(description="Launch worker")
parser.add_argument("--name", type=str, default="50000")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=str, default="50000")
parser.add_argument("--verbose", action="store_true")
parser.add_argument(
    "--cluster-spec", default="moose/cluster/cluster-spec-docker-compose.yaml"
)
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

if __name__ == "__main__":
    aio.init_grpc_aio()

    clusters_spec = load_cluster_spec(args.cluster_spec)
    channel_manager = ChannelManager(clusters_spec)
    executor = KernelBasedExecutor(name="remote", channel_manager=channel_manager)

    get_logger().info(f"Starting on {args.host}:{args.port}")
    server = Server(args.host, args.port, executor)

    asyncio.get_event_loop().run_until_complete(server.start())
    get_logger().info("Started")

    asyncio.get_event_loop().run_until_complete(server.wait())
    get_logger().info("Stopped")
