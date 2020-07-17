import logging 
import asyncio 

from channels import ChannelServer


if __name__ == "__main__":
    logging.basicConfig()
    loop = asyncio.get_event_loop() 
    channel_server = ChannelServer("localhost", "50051")                                              
    loop.create_task(channel_server.start())                                       
    loop.run_forever()


