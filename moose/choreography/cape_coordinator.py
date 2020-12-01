import asyncio
import base64
import itertools
import socket

from cape.api.session import ComputationStatus
from cape.api.session import Session
from cape.cape import Cape

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
        self.cape = Cape()
        # self.client = Client(coordinator_host, auth_token)
        self.executor = executor
        self.own_name = own_name or socket.gethostname()
        self.poll_delay = poll_delay
        self.session_tasks = dict()

    async def _handle_session(
        self, session: Session, computation, placement_instantiation, placement,
    ):
        get_logger().debug(f"Handling new session; session_id:{session.id}")
        await self._report_session_status(session, ComputationStatus.Started)
        get_logger().debug(f"Starting execution; session_id:{session.id}")
        try:
            await self.executor.run_computation(
                logical_computation=computation,
                placement_instantiation=placement_instantiation,
                placement=placement,
                session_id=session.id,
            )
        except Exception as ex:
            get_logger().error(
                f"Error occured during execution; session_id:{session.id}, ex:{ex}"
            )
            await self._report_session_status(session, ComputationStatus.Error)
            return

        get_logger().debug(f"Finished execution; session_id:{session.id}")
        await self._report_session_status(session, ComputationStatus.Completed)

    async def _get_next_sessions(self):
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self.cape.next_sessions)
        except Exception as ex:
            get_logger().error(f"Failed getting next sessions; ex:{ex}")

    async def _report_session_status(self, session, status):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, self.cape.report_session_status, session, status,
            )
            get_logger().debug("Reported successfully")
        except Exception as ex:
            get_logger().error(
                f"Failed to report session status;"
                f" session_id:{session.id},"
                f" status:{status},"
                f" ex:{ex}"
            )

    # async def _login(self):
    #     loop = asyncio.get_event_loop()
    #     await loop.run_in_executor(
    #         None, self.client.login,
    #     )
    #     get_logger().debug("Logged in successfully")

    async def run(self):
        # await self._login()
        for i in itertools.count(start=0):
            if i > 0:
                await asyncio.sleep(self.poll_delay)

            sessions = await self._get_next_sessions()
            for session in sessions:
                session_id = session.id
                if session_id in self.session_tasks:
                    get_logger().debug(
                        f"Ignoring session since it already exists;"
                        f" session_id:{session_id}"
                    )
                    continue

                placement_instantiation = session.placement_instantiation
                placement = None  # TODO we should receive a placement as well
                computation_bytes = base64.b64decode(session.task.computation)
                computation = Computation.deserialize(computation_bytes)

                task = asyncio.create_task(
                    self._handle_session(
                        session=session,
                        computation=computation,
                        placement_instantiation=placement_instantiation,
                        placement=placement,
                    )
                )
                self.session_tasks[session.id] = task

            get_logger().debug(f"Saw {len(sessions)} sessions")
