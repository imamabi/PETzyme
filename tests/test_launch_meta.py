import os
import subprocess

import pytest

from scripts import launch_meta


def test_no_authorization_does_not_access_keychain(monkeypatch):
    monkeypatch.setattr(launch_meta, '_lookup', lambda _: pytest.fail('Keychain accessed'))
    with pytest.raises(SystemExit) as error:
        launch_meta.main([])
    assert error.value.code == 2


@pytest.mark.parametrize('returncode,expected', [(0, True), (44, False)])
def test_presence_lookup_never_reads_password(monkeypatch, capsys, returncode, expected):
    calls = []
    monkeypatch.setattr(launch_meta.sys, 'platform', 'darwin')
    monkeypatch.setenv('MODEL_API_KEY', 'SYNTHETIC_AMBIENT')
    def run(command, **kwargs):
        calls.append(command)
        assert command[:2] == ['/usr/bin/security', 'find-generic-password']
        assert command[2:6] == ['-s', 'zymecycle-meta-api', '-a', 'zymecycle']
        assert command[-1].endswith('/Library/Keychains/login.keychain-db')
        assert '-w' not in command and '-g' not in command
        assert kwargs['stdout'] == kwargs['stderr'] == subprocess.DEVNULL
        assert 'MODEL_API_KEY' not in kwargs['env']
        return subprocess.CompletedProcess(command, returncode)
    monkeypatch.setattr(launch_meta.subprocess, 'run', run)
    assert launch_meta.main(['--check-keychain']) == (0 if expected else 1)
    assert capsys.readouterr().out == f'Keychain item present: {str(expected).lower()}\n'
    assert len(calls) == 1


def test_live_child_only_receives_key_in_environment(monkeypatch, capsys):
    monkeypatch.setattr(launch_meta.sys, 'platform', 'darwin')
    monkeypatch.setenv('MODEL_API_KEY', 'SYNTHETIC_AMBIENT')
    original = dict(os.environ)
    calls = []
    child_environments = []
    def run(command, **kwargs):
        calls.append(command)
        assert 'shell' not in kwargs and 'SYNTHETIC_KEYCHAIN' not in command
        if command[0] == '/usr/bin/security':
            assert '-w' in command and 'MODEL_API_KEY' not in kwargs['env']
            assert kwargs['stdout'] == subprocess.PIPE and kwargs['stderr'] == subprocess.DEVNULL
            return subprocess.CompletedProcess(command, 0, stdout=b'SYNTHETIC_KEYCHAIN\n')
        assert kwargs['env']['MODEL_API_KEY'] == 'SYNTHETIC_KEYCHAIN'
        child_environments.append(kwargs['env'])
        return subprocess.CompletedProcess(command, 7)
    monkeypatch.setattr(launch_meta.subprocess, 'run', run)
    assert launch_meta.main(['--authorize-paid-meta', '--', 'python', '-m', 'scripts.research_agent', 'question', '--authorize-paid-meta']) == 7
    assert len(calls) == 2 and calls[1][0] == 'python'
    assert os.environ == original
    assert 'MODEL_API_KEY' not in child_environments[0]
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('output', [None, b'', b'bad\nheader', b'bad key', b'\xff'])
def test_missing_or_invalid_credential_never_launches_child(monkeypatch, capsys, output):
    result = None if output is None else subprocess.CompletedProcess([], 0, stdout=output)
    monkeypatch.setattr(launch_meta, '_lookup', lambda _: result)
    monkeypatch.setattr(launch_meta.subprocess, 'run', lambda *a, **k: pytest.fail('Child launched'))
    assert launch_meta.launch_live(['python', '-m', 'scripts.research_agent']) == 1
    assert capsys.readouterr().err == 'Keychain credential unavailable; no live child was launched.\n'


@pytest.mark.parametrize('error', [OSError('PRIVATE'), subprocess.TimeoutExpired('security', 60, output=b'PRIVATE')])
def test_security_errors_are_sanitized(monkeypatch, capsys, error):
    monkeypatch.setattr(launch_meta.sys, 'platform', 'darwin')
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(launch_meta.subprocess, 'run', fail)
    assert launch_meta.launch_live(['python']) == 1
    assert 'PRIVATE' not in str(capsys.readouterr())


def test_import_and_offline_cli_do_not_access_keychain(monkeypatch):
    from scripts.research_agent import main
    monkeypatch.setattr(launch_meta.subprocess, 'run', lambda *a, **k: pytest.fail('Keychain accessed'))
    assert main(['Compare ESM050 at 60°C and 40°C, pH 6.5, powder']) == 1


def test_presence_mode_rejects_child_before_lookup(monkeypatch):
    monkeypatch.setattr(launch_meta, '_lookup', lambda _: pytest.fail('Keychain accessed'))
    with pytest.raises(SystemExit):
        launch_meta.main(['--check-keychain', '--', 'python'])
