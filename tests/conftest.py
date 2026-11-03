import socket
import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PETZYME_OFFLINE", "1")
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
