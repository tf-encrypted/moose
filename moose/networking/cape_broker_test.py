import pysodium
import responses

from moose.networking.cape_broker import Networking

coor_host = "http://localhost:8080"
broker_host = "http://localhost:5050"
token = "01EQXZ0FKYNF8MX4MJZPWQ4KTA,Ae-gaS4Ey_Hz9eq_MvGJigiwx4coy8vDmA"


class _receiver:
    def __init__(self, public_key, endpoint):
        self.public_key = public_key
        self.endpoint = endpoint


class _sender:
    def __init__(self, public_key, endpoint):
        self.public_key = public_key
        self.endpoint = endpoint


@responses.activate
def test_post(event_loop):
    public_key, secret_key = pysodium.crypto_box_keypair()

    exp_token = "ABCD"
    key = "key"
    id = "id"
    responses.add(
        responses.POST, f"{coor_host}/v1/login", json={"token": exp_token},
    )
    responses.add(responses.POST, f"{broker_host}/{id}/{key}")

    n = Networking(
        "http://localhost:5050",
        "http://localhost:8080",
        auth_token=token,
        public_key=public_key,
        secret_key=secret_key,
    )

    event_loop.run_until_complete(
        n.send("value", None, _receiver(public_key, ""), key, id)
    )


@responses.activate
def test_get(event_loop):
    public_key, secret_key = pysodium.crypto_box_keypair()

    exp_token = "ABCD"
    key = "key"
    id = "id"
    value = "value"
    responses.add(
        responses.POST, f"{coor_host}/v1/login", json={"token": exp_token},
    )
    responses.add(
        responses.GET, f"{broker_host}/{id}/{key}", body=value,
    )

    n = Networking(
        "http://localhost:5050",
        "http://localhost:8080",
        auth_token=token,
        public_key=public_key,
        secret_key=secret_key,
    )

    res = event_loop.run_until_complete(
        n.receive(
            _sender(None, "http://localhost:5050"),
            _receiver(None, "http://localhost:5050"),
            key,
            id,
        )
    )

    assert res.decode("ascii") == value
