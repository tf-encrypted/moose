import click

from cape.network.client import Client
from cape.vars import token


@click.group()
def tasks():
    pass


@click.command('create')
@click.argument('project_id')
@click.argument('type')
def tasks_create(project_id, type):
    client = Client('http://localhost:8080', token)
    client.login()

    resp = client.create_task(project_id, type)
    print('res', resp)

    # click.echo('made project: ' + name)


@click.command('assign')
@click.argument('task_id')
@click.argument('task_roles') # type: Dict[str, str]):
def tasks_assign(task_id, task_roles):
    print('wow!', task_id, task_roles, type(task_roles))

    client = Client('http://localhost:8080', token)
    client.login()

    resp = client.assign_task_roles(task_id, task_roles)
    print(resp)


tasks.add_command(tasks_create)
tasks.add_command(tasks_assign)
