import asyncio
import requests


class Networking:
    def __init__(self, broker_host):
        self.broker_host = broker_host
        self.session = requests.Session()

    def get_hostname(self, placement):
        endpoint = placement
        host, port = endpoint.split(":")
        return host

    async def _get(self, endpoint):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            None,
            self.session.get,
            f"{self.broker_host}/{endpoint}",
        )

    async def _put(self, endpoint, value):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            None,
            self.session.put,
            f"{self.broker_host}/{endpoint}",
            data=value,
        )

    async def receive(self, sender, receiver, rendezvous_key, session_id):
        return await self._get(f"{session_id}/{rendezvous_key}")

    async def send(self, value, sender, receiver, rendezvous_key, session_id):
        await self._put(f"{session_id}/{rendezvous_key}", value)
