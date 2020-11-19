import click


from cape.projects import projects
from cape.sessions import sessions
from cape.tasks import tasks


@click.group()
def cli():
    pass


cli.add_command(projects)
cli.add_command(tasks)
cli.add_command(sessions)

if __name__ == '__main__':
    cli()
