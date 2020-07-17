import asyncio
from collections import defaultdict


class AsyncMemoryChannel:
    def __init__(self):
        # TODO(Morten) having an async dict-like structure
        # would probably be better ...
        self.queues = defaultdict(asyncio.Queue)

    async def send(self, value, rendezvous_key, session_id):
        queue_key = (session_id, rendezvous_key)
        queue = self.queues[queue_key]
        return await queue.put(value)

    async def receive(self, rendezvous_key, session_id):
        queue_key = (session_id, rendezvous_key)
        queue = self.queues[queue_key]
        return await queue.get()


class AsyncChannelManager:
    def __init__(self):
        self.channels = defaultdict(AsyncMemoryChannel)

    def get_channel(self, op):
        channel_key = (op.sender, op.receiver)
        return self.channels[channel_key]

    async def send(self, value, op, session_id):
        await self.get_channel(op).send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    async def receive(self, op, session_id):
        return await self.get_channel(op).receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )
