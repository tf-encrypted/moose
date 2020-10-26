import argparse
import asyncio
import logging

from moose.cluster.cluster_spec import load_cluster_spec
from moose.logger import get_logger
from moose.worker import Worker

parser = argparse.ArgumentParser(description="Launch worker")
parser.add_argument("--name", type=str, default="Worker")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=str, default="50000")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--cluster-spec", default="cluster-spec.yaml")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

if __name__ == "__main__":
    get_logger().info(f"Starting on {args.host}:{args.port}")
    cluster_spec = load_cluster_spec(args.cluster_spec)
    worker = Worker(args.name, args.host, args.port, cluster_spec)

    asyncio.get_event_loop().run_until_complete(worker.start())
    get_logger().info("Started")

    asyncio.get_event_loop().run_until_complete(worker.wait())
    get_logger().info("Stopped")
