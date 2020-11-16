import asyncio

import requests

from moose.logger import get_logger


class Networking:
    def __init__(self, broker_host):
        self.broker_host = broker_host
        self.session = requests.Session()

    def get_hostname(self, placement):
        endpoint = placement
        host, port = endpoint.split(":")
        return host

    async def _get(self, endpoint, delay=1.0, max_attempts=60):
        loop = asyncio.get_event_loop()
        for i in range(max_attempts):
            if i > 0:
                await asyncio.sleep(delay)
            try:
                res = await loop.run_in_executor(None, self.session.get, endpoint)
            except requests.exceptions.ConnectionError:
                continue
            if res.status_code == requests.codes.ok:
                get_logger().debug(f"GET success; endpoint:'{endpoint}', attempts:{i}")
                return res.content
            if res.status_code == requests.codes.not_found:
                continue
            get_logger().error(
                f"GET unhandled error:"
                f" endpoint:'{endpoint}',"
                f" status_code:{res.status_code}"
            )
        get_logger().error(
            f"GET failure: max attempts reached;"
            f" endpoint:'{endpoint}',"
            f" attempts:{i}"
        )
        raise IOError()

    async def _post(self, endpoint, value, delay=1.0, max_attempts=60):
        loop = asyncio.get_event_loop()
        for i in range(max_attempts):
            if i > 0:
                await asyncio.sleep(delay)
            res = await loop.run_in_executor(None, self.session.post, endpoint, value)
            if res.status_code == requests.codes.ok:
                get_logger().debug(f"POST success; endpoint:'{endpoint}', attempts:{i}")
                return
            get_logger().error(
                f"POST unhandled error:"
                f" endpoint:'{endpoint}',"
                f" status_code:{res.status_code}"
            )
        get_logger().error(
            f"POST failure: max attempts reached;"
            f" endpoint:'{endpoint}',"
            f" max_attempts:{max_attempts}"
        )
        raise IOError()

    async def receive(self, sender, receiver, rendezvous_key, session_id):
        return await self._get(
            f"http://{self.broker_host}/{session_id}/{rendezvous_key}"
        )

    async def send(self, value, sender, receiver, rendezvous_key, session_id):
        await self._post(
            f"http://{self.broker_host}/{session_id}/{rendezvous_key}", value
        )
