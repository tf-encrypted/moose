from pprint import pprint
from typing import Dict

import requests

from cape.network import base64
from cape.network.api_token import APIToken


class Client:
    def __init__(self, endpoint: str, token: str):
        self.endpoint = endpoint
        self.gql_endpoint = self.endpoint + '/v1/query'
        self.api_token = APIToken(token)
        self.session = requests.Session()
        self.token = ''

    def login(self):
        resp = self.session.post(self.endpoint + '/v1/login', json={
            'token_id': self.api_token.token_id,
            'secret': self.api_token.secret
        })

        json = resp.json()
        self.token = base64.from_string(json['token'])

    def gql_req(self, query, variables):
        r = self.session.post(self.gql_endpoint, json={
            'query': query,
            'variables': variables
        })

        return pprint(r.json())

    def create_project(self, name):
        return self.gql_req("""
            mutation CreateProject($project: CreateProjectRequest!) {
                createProject(project: $project) {
                    id
                }
            }
            """, {
            'project': {
                'name': name,
                'Description': 'who cares lets get rid of this field'
            }
        })

    def list_projects(self):
        return self.gql_req("""
            query GetProjects($status: ProjectStatus!) {
                projects(status: $status) {
                    id
                    label
                    name
                }
            }
            """, {
            'status': 'any'
        })

    def create_task(self, project_id, task_type):
        return self.gql_req("""
            mutation CreateTask($project_id: String!, $task_type: TaskType!) {
                createTask(project_id: $project_id, task_type: $task_type) {
                    id
                }
            }
        """, {
            'project_id': project_id,
            'task_type': task_type
        })

    def assign_task_roles(self, task_id: str, task_roles):
        return self.gql_req("""
            mutation AssignTaskRoles($task_id: String!, $task_roles: TaskRolesInput!) {
                assignTaskRoles(task_id: $task_id, task_roles: $task_roles) {
                    id
                }
            }
        """, {
            'task_id': task_id,
            'task_roles': task_roles
        })

    def initialize_session(self, task_id):
        return self.gql_req("""
            mutation InitializeSession($task_id: String!) {
                initializeSession(task_id: $task_id) {
                    id
                    placementInstantiation {
                        label
                        endpoint
                    }
                }
            }
            """, {
                'task_id': task_id
        })

    def get_next_sessions(self):
        query = """
        query GetNextSessions($workerName: String!) {
            getNextSessions(workerName: $workerName) {
                id
                placementInstantiation {
                    label
                    endpoint
                }
            }
        }
        """

        variables = {'workerName': 'inputter0'}
        r = self.session.post(self.gql_endpoint, json={
            'query': query,
            'variables': variables
        })

        return r.json()['data']['getNextSessions']
