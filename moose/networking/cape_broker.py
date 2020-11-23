import asyncio
import socket

import pysodium
import requests
from opentelemetry import trace
from opentelemetry.launcher import configure_opentelemetry

from moose.logger import get_logger


class Networking:
    def __init__(self, broker_host, own_name=None, auth_token=None):
        self.broker_host = broker_host
        self.session = requests.Session()
        # TODO(Morten) how should we authenticate?
        self.session.auth = (own_name or socket.gethostname(), auth_token or "")
        self._my_public_key = None
        self._my_secret_key = None

    def get_hostname(self, placement):
        endpoint = placement
        host, port = endpoint.split(":")
        return host

    def _get_wrapper(self, endpoint):
        return self.session.get(url=endpoint)

    def _post_wrapper(self, endpoint, value):
        return self.session.post(url=endpoint, data=value)

    async def _get(self, endpoint, delay=1.0, max_attempts=60):
        loop = asyncio.get_event_loop()
        for i in range(max_attempts):
            if i > 0:
                await asyncio.sleep(delay)
            try:
                res = await loop.run_in_executor(None, self._get_wrapper, endpoint)
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
        ex = Exception(
            f"GET failure: max attempts reached;"
            f" endpoint:'{endpoint}',"
            f" attempts:{i}"
        )
        get_logger().exception(ex)
        raise ex

    async def _post(self, endpoint, value, delay=1.0, max_attempts=60):
        loop = asyncio.get_event_loop()
        for i in range(max_attempts):
            if i > 0:
                await asyncio.sleep(delay)
            res = await loop.run_in_executor(None, self._post_wrapper, endpoint, value)
            if res.status_code == requests.codes.ok:
                get_logger().debug(f"POST success; endpoint:'{endpoint}', attempts:{i}")
                return
            get_logger().error(
                f"POST unhandled error:"
                f" endpoint:'{endpoint}',"
                f" status_code:{res.status_code}"
            )
        ex = Exception(
            f"POST failure: max attempts reached;"
            f" endpoint:'{endpoint}',"
            f" max_attempts:{max_attempts}"
        )
        get_logger().exception(ex)
        raise ex

    async def receive(self, sender, receiver, rendezvous_key, session_id):
        encrypted_value = await self._get(f"{self.broker_host}/{session_id}/{rendezvous_key}")
        # Decrypt value with libsodium
        # run an hash function on it to generate the nonce.
        n = bytes(rendezvous_key + session_id)
        sender_public_key = self._other_parties_public_key[sender]
        decrypted_value = pysodium.crypto_box_open(encrypted_value, n, sender_public_key, self._my_secret_key)
        return decrypted_value

    async def send(self, value, sender, receiver, rendezvous_key, session_id):
        # Encrypt value with libsodium
        n = bytes(rendezvous_key + session_id)
        receiver_public_key = self._other_parties_public_key[receiver]
        encrypted_value = pysodium.crypto_box(value, nonce, receiver_public_key, self._my_secret_key)
        await self._post(f"{self.broker_host}/{session_id}/{rendezvous_key}", encrypted_value)

    def setup_keys(self):
        self._my_public_key, self._my_secret_key = pysodium.crypto_box_keypair()


class TelemetryNetworking(Networking):
    def __init__(self, broker_host, own_name=None, auth_token=None):
        configure_opentelemetry(service_name="worker-broker-client")

        self.tracer = trace.get_tracer(__name__)
        super().__init__(broker_host, own_name, auth_token)

    def _get_wrapper(self, endpoint):
        with self.tracer.start_as_current_span("get"):
            return super().session.get(url=endpoint)

    def _post_wrapper(self, endpoint, value):
        with self.tracer.start_as_current_span("post"):
            return super().session.post(url=endpoint, data=value)


def get_networking(broker_host, own_name=None, auth_token=None, telemetry_enable=False):
    if telemetry_enable:
        return TelemetryNetworking(
            broker_host, own_name=own_name, auth_token=auth_token
        )
