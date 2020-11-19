import asyncio
import functools
import itertools
import random
import socket
from typing import Dict

import requests

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
        self.executor = executor
        self.coordinator_host = coordinator_host
        self.own_name = own_name or socket.gethostname()
        self.requests_session = requests.Session()
        self.session_tasks = dict()
        self.poll_delay = poll_delay

    async def graphql_request(self, query, variables):
        loop = asyncio.get_event_loop()
        r = await loop.run_in_executor(
            None,
            functools.partial(
                func=self.requests_session.post,
                url=f"{self.coordinator_host}/v1/query",
                json={"query": query, "variables": variables},
            ),
        )
        try:
            j = r.json()
        except ValueError:
            r.raise_for_status()

        if "errors" in j:
            raise Exception(j["errors"])

        return j["data"]

    def launch_session(
        self, session_id, computation, placement_instantiation, placement
    ):
        if session_id in self.session_tasks:
            get_logger().debug(
                f"Ignoring session since it already exists; session_id:{session_id}"
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
        query = """
            query GetNextSessions($workerName: String!) {
                getNextSessions(workerName: $workerName) {
                    id
                    computation {
                        computation
                    }
                    placementInstantiation {
                        label
                        endpoint
                    }
                    status
                }
            }
        """
        variables = {"workerName": self.own_name}
        res = await self.graphql_request(query, variables)
        get_logger().debug(res)

    async def run():
        for i in itertools.count(start=1):
            if i > 0:
                await asyncio.sleep(self.poll_delay)
            sessions = await self.poll()
            # TODO(Morten) launch sessions
