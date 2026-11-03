"""Synthetic setup tests: no remote service or real source records are used."""
from __future__ import annotations

import hashlib

import httpx
import pytest

from scripts import prepare_pazy_activity as setup


def test_preparation_validates_and_reuses_local_archive(tmp_path, monkeypatch):
    content = b'synthetic-compressed-fixture'
    calls = []
    monkeypatch.setattr(setup, 'SHA256', hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(setup, 'load_pazy_activity_snapshot', lambda path: {'status': 'SUCCESS'})
    client = httpx.Client
    def respond(request):
        calls.append(request.url)
        return httpx.Response(200, content=content)
    monkeypatch.setattr(setup.httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(respond)))
    assert setup.prepare_snapshot(tmp_path)['reused'] is False
    assert setup.prepare_snapshot(tmp_path)['reused'] is True
    assert len(calls) == 1 and calls[0].scheme == 'https'


def test_bad_checksum_never_replaces_or_retains_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, 'SHA256', '0' * 64)
    client = httpx.Client
    monkeypatch.setattr(setup.httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b'wrong'))))
    with pytest.raises(ValueError, match='checksum'):
        setup.prepare_snapshot(tmp_path)
    directory = tmp_path / 'data' / 'pazy'
    assert not (directory / setup.FILENAME).exists()
    assert not (directory / (setup.FILENAME + '.part')).exists()


def test_preexisting_partial_is_never_removed(tmp_path, monkeypatch):
    directory = tmp_path / 'data' / 'pazy'
    directory.mkdir(parents=True)
    partial = directory / (setup.FILENAME + '.part')
    partial.write_bytes(b'existing')
    client = httpx.Client
    monkeypatch.setattr(setup.httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b'new'))))
    with pytest.raises(FileExistsError):
        setup.prepare_snapshot(tmp_path)
    assert partial.read_bytes() == b'existing'
