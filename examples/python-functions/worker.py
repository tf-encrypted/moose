import argparse
import asyncio
import logging
import os

from moose.logger import get_logger
from moose.worker import Worker

parser = argparse.ArgumentParser(description="Launch worker")
parser.add_argument("--name", type=str, default="Worker")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=str, default="50000")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--ca-cert", default=os.environ.get("CA_CERT", None))
parser.add_argument("--ident-cert", default=os.environ.get("IDENT_CERT", None))
parser.add_argument("--ident-key", default=os.environ.get("IDENT_KEY", None))
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)

if __name__ == "__main__":

    worker = Worker(
        name=args.name,
        host=args.host,
        port=args.port,
        ca_cert_filename=args.ca_cert,
        ident_cert_filename=args.ident_cert,
        ident_key_filename=args.ident_key,
        allow_insecure_networking=True,
    )

    asyncio.get_event_loop().run_until_complete(worker.start())
    get_logger().info("Started")

    asyncio.get_event_loop().run_until_complete(worker.wait())
    get_logger().info("Stopped")
