import logging 
import asyncio 

from channels import ChannelServer

from grpc.experimental import aio 


if __name__ == "__main__":
    logging.basicConfig()
    aio.init_grpc_aio()

    loop = asyncio.get_event_loop() 

    channel_server = ChannelServer("localhost", "50052")
    loop.run_until_complete(channel_server.start())
    loop.run_until_complete(channel_server.wait())
    print("Done")
