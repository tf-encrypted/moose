import requests

from cape import base64
from cape.api_token import create_api_token, APIToken


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

        print('logged in w/ token', self.token)

    def get_next_sessions(self):
        query = """
        query GetNextSessions($workerName: String!) {
          getNextSessions(workerName: $workerName) {
            id
            computation { # this block is currently changing ...
              id
              name
              computation # this is the json definition of the comp
            }
            placementInstantiation {
              label
              endpoint
            }
            status
          }
        }
        """

        variables = {'workerName': 'inputter0'}
        r = self.session.post(self.gql_endpoint, json={
            'query': query,
            'variables': variables
        })

        print('done', r.json())