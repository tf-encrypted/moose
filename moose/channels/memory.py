from collections import defaultdict

from moose.storage import AsyncStore


class Channel:
    def __init__(self):
        self.buffer = AsyncStore()

    async def receive(self, rendezvous_key, session_id):
        key = (session_id, rendezvous_key)
        return await self.buffer.get(key)

    async def send(self, value, rendezvous_key, session_id):
        key = (session_id, rendezvous_key)
        return await self.buffer.put(key, value)


class ChannelManager:
    def __init__(self):
        self.channels = defaultdict(Channel)

    def get_channel(self, op):
        channel_key = (op.sender, op.receiver)
        return self.channels[channel_key]

    async def receive(self, op, session_id):
        return await self.get_channel(op).receive(
            rendezvous_key=op.rendezvous_key, session_id=session_id
        )

    async def send(self, value, op, session_id):
        await self.get_channel(op).send(
            value, rendezvous_key=op.rendezvous_key, session_id=session_id
        )
