from collections import defaultdict

from moose.logger import get_meter
from moose.utils import AsyncStore


class Channel:
    def __init__(self):
        self.buffer = AsyncStore()

    async def receive(self, rendezvous_key, session_id):
        key = (session_id, rendezvous_key)
        return await self.buffer.get(key)

    async def send(self, value, rendezvous_key, session_id):
        key = (session_id, rendezvous_key)
        return await self.buffer.put(key, value)


class Networking:
    def __init__(self):
        self.channels = defaultdict(Channel)
        self.sent_counter = get_meter().create_counter(
            name="bytes_sent",
            description="number of bytes sent",
            unit="bytes",
            value_type=int,
        )
        self.received_counter = get_meter().create_counter(
            name="bytes_received",
            description="number of bytes received",
            unit="bytes",
            value_type=int,
        )

    def get_hostname(self, placement):
        return "localhost"

    async def receive(self, sender, receiver, rendezvous_key, session_id):
        channel_key = (sender, receiver)
        value = await self.channels[channel_key].receive(
            rendezvous_key=rendezvous_key, session_id=session_id
        )
        self.received_counter.add(
            len(value) * 8, labels={"moose.session_id": session_id}
        )
        return value

    async def send(self, value, sender, receiver, rendezvous_key, session_id):
        channel_key = (sender, receiver)
        self.sent_counter.add(len(value) * 8, labels={"moose.session_id": session_id})
        await self.channels[channel_key].send(
            value, rendezvous_key=rendezvous_key, session_id=session_id
        )
