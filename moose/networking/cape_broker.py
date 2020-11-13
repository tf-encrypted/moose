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
            get_logger().debug(f"Getting value: endpoint:'{endpoint}'")
            try:
                res = await loop.run_in_executor(None, self.session.get, endpoint,)
            except requests.exceptions.ConnectionError:
                get_logger().debug(f"Connection not ready yet: endpoint:'{endpoint}'")
                continue
            if res.status_code == requests.codes.ok:
                return res
            if res.status_code == requests.codes.not_found:
                get_logger().debug(f"Value not ready yet: endpoint:'{endpoint}'")
                continue
            get_logger().error(
                f"Unknown error getting value:"
                f" endpoint:'{endpoint}',"
                f" status_code:{res.status_code}"
            )
        get_logger().error(
            f"Max attempts reached getting value:"
            f" endpoint:'{endpoint}',"
            f" max_attempts:{max_attempts}"
        )
        raise IOError()

    async def _put(self, endpoint, value, delay=1.0, max_attempts=60):
        loop = asyncio.get_event_loop()
        for i in range(max_attempts):
            if i > 0:
                await asyncio.sleep(delay)
            get_logger().debug(f"Putting value: endpoint:'{endpoint}'")
            res = await loop.run_in_executor(
                None, self.session.put, endpoint, data=value,
            )
            if res.status_code == requests.codes.ok:
                return
            else:
                get_logger().error(
                    f"Unknown error putting value:"
                    f" endpoint:'{endpoint}',"
                    f" status_code:{res.status_code}"
                )
        get_logger().error(
            f"Max attempts reached putting value:"
            f" endpoint:'{endpoint}',"
            f" max_attempts:{max_attempts}"
        )
        raise IOError()

    async def receive(self, sender, receiver, rendezvous_key, session_id):
        return await self._get(
            f"http://{self.broker_host}/{session_id}/{rendezvous_key}"
        )

    async def send(self, value, sender, receiver, rendezvous_key, session_id):
        await self._put(
            f"http://{self.broker_host}/{session_id}/{rendezvous_key}", value
        )
