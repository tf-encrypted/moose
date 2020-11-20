import asyncio
import functools
import itertools
import random
import socket
from pprint import pprint
from typing import Dict

from cape.network.client import Client
from moose.compiler.computation import Computation
from moose.logger import get_logger


class Choreography:
    def __init__(
        self,
        executor,
        coordinator_host,
        own_name=None,
        auth_token=None,
        poll_delay=10.0,
    ):
        self.client = Client(coordinator_host, auth_token)
        self.executor = executor
        self.own_name = own_name or socket.gethostname()
        self.poll_delay = poll_delay
        self.session_tasks = dict()

    def launch_session(
        self, session_id, computation, placement_instantiation, placement
    ):
        if session_id in self.session_tasks:
            get_logger().debug(
                f"Ignoring session since it already exists;"
                f" session_id:{session_id}"
            )
            return
        task = asyncio.create_task(
            self.executor.run_computation(
                logical_computation=Computation.deserialize(computation),
                placement_instantiation=placement_instantiation,
                placement=placement,
                session_id=session_id,
            )
        )
        self.session_tasks[session_id] = task
        get_logger().debug(f"Launched new computation; session_id:{session_id}")

    async def poll(self):
        loop = asyncio.get_event_loop()
        sessions = await loop.run_in_executor(
            None, self.client.get_next_sessions, self.own_name,
        )
        get_logger().debug(f"Polled sessions; sessions:{sessions}")
        return sessions

    async def login(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.client.login,
        )
        get_logger().debug("Logged in successfullly")

    async def run(self):
        await self.login()
        for i in itertools.count(start=1):
            if i > 0:
                await asyncio.sleep(self.poll_delay)
            sessions = await self.poll()
            # TODO(Morten) do something with sessions
