import click

from cape.network.client import Client
from cape.vars import token


@click.group()
def projects():
    pass


@click.command("create")
@click.argument("name")
def project_create(name):
    client = Client("http://localhost:8080", token)
    client.login()

    resp = client.create_project(name)
    print("res", resp)

    # click.echo('made project: ' + name)


@click.command("list")
def project_list():
    client = Client("http://localhost:8080", token)
    client.login()

    resp = client.list_projects()
    print("res", resp)


projects.add_command(project_list)
projects.add_command(project_create)
