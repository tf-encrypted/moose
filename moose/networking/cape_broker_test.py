import pytest
import responses

from moose.networking.cape_broker import Networking

coor_host = "http://localhost:8080"
broker_host = "http://localhost:5050"
token = "01EQXZ0FKYNF8MX4MJZPWQ4KTA,Ae-gaS4Ey_Hz9eq_MvGJigiwx4coy8vDmA"

@responses.activate
def test_post(event_loop):
    exp_token = "ABCD"
    key = "key"
    id = "id"
    responses.add(
        responses.POST,
        f"{coor_host}/v1/login",
        json={"token": exp_token},
    )
    responses.add(
        responses.POST,
        f"{broker_host}/{id}/{key}"
    )

    n = Networking("http://localhost:5050", "http://localhost:8080", auth_token=token)

    event_loop.run_until_complete(n.send("value", None, None, key, id))


@responses.activate
def test_get(event_loop):
    exp_token = "ABCD"
    key = "key"
    id = "id"
    responses.add(
        responses.POST,
        f"{coor_host}/v1/login",
        json={"token": exp_token},
    )
    responses.add(
        responses.GET,
        f"{broker_host}/{id}/{key}",
        body="value",
    )

    n = Networking("http://localhost:5050", "http://localhost:8080", auth_token=token)

    event_loop.run_until_complete(n.receive(None, None, key, id))