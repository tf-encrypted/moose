import argparse
import logging
import os
import uuid

from cape.api.projects import Project
from cape.api.task import MultiplicationTask
from cape.cape import Cape

from moose.logger import get_logger

parser = argparse.ArgumentParser(description="Seed Computation Session")
parser.add_argument("--token_alice", default=os.environ.get("CAPE_TOKEN_ALICE"))
parser.add_argument("--token_bob", default=os.environ.get("CAPE_TOKEN_BOB"))
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

logger = logging.getLogger("cape-utils")
if not args.quiet:
    get_logger().setLevel(level=logging.DEBUG)
    logger.setLevel(level=logging.DEBUG)

if __name__ == "__main__":
    if args.token_alice is None or args.token_bob is None:
        logger.error(
            "set a token for workers with --token_alice & --token_bob .. see "
            "https://app.clickup.com/8456309/v/dc/8223n-1078/8223n-287"
        )
        quit()

    logger.info(
        "seeding cape comp session w/ "
        f"alice_token: {args.token_alice} and token_bob: {args.token_bob}"
    )
    logger.info("logging into cape")
    cape = Cape(token=args.token_alice)
    project = Project(
        name=f"my-project-{str(uuid.uuid4())[:8]}",
        description="Made by the session test",
    )

    worker_email = "bob@capeprivacy.com"

    logger.info("getting workers")
    alice = cape.get_user("alice@capeprivacy.com")
    bob = cape.get_user("bob@capeprivacy.com")

    logger.info(f"creating project {project.name}")
    p = cape.create_project(project)

    logger.info("adding worker bob to the project")
    c_id = cape.add_member(p, "bob@capeprivacy.com")

    task = MultiplicationTask()

    logger.info("adding multiplication task to the project")
    task = cape.add_task(p, task)

    logger.info(f"assigning task roles to task {task.id}")
    task = cape.assign_task_roles(task, {"inputter0": alice.id, "inputter1": bob.id})

    logger.info(f"initializing session for task {task.id}")
    session = cape.initialize_session(task)
    logger.info(f"initialized session {session.id}")
