import click

from cape.network.client import Client
from cape.vars import token


@click.group()
def sessions():
    pass


@click.command("create")
@click.argument("task_id")
def sessions_create(task_id):
    client = Client("http://localhost:8080", token)
    client.login()

    resp = client.initialize_session(task_id)
    print("res", resp)


sessions.add_command(sessions_create)
