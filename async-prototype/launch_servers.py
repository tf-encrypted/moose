import argparse
import logging
import asyncio
from grpc.experimental import aio

from channels import Server

parser = argparse.ArgumentParser(description="Launch servers")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=str, default="50051")
args = parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig()
    aio.init_grpc_aio()

    server = Server(args.host, args.port)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.start())
    loop.run_until_complete(server.wait())
