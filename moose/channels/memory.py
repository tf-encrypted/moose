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

    def get_hostname(self, player_name):
        return "localhost"

    async def receive(self, sender, receiver, rendezvous_key, session_id):
        channel_key = (sender, receiver)
        return await self.channels[channel_key].receive(
            rendezvous_key=rendezvous_key, session_id=session_id
        )

    async def send(self, value, sender, receiver, rendezvous_key, session_id):
        channel_key = (sender, receiver)
        await self.channels[channel_key].send(
            value, rendezvous_key=rendezvous_key, session_id=session_id
        )
