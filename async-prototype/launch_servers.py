import logging 
import asyncio 

from grpc.experimental import aio 

from channels import ChannelServer

if __name__ == "__main__":
    logging.basicConfig()
    aio.init_grpc_aio()

    loop = asyncio.get_event_loop()

    channel_server = ChannelServer("localhost", "50051")
    loop.run_until_complete(channel_server.start())
    loop.run_until_complete(channel_server.wait())
    print("Done")
