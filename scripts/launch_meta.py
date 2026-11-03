"""Launch one authorized child using Keychain or a process environment credential."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SERVICE = 'zymecycle-meta-api'
ACCOUNT = 'zymecycle'


def _environment() -> dict[str, str]:
    """Copy the environment without forwarding an ambient Meta credential."""
    return {key: value for key, value in os.environ.items() if key != 'MODEL_API_KEY'}


def _lookup(password: bool) -> subprocess.CompletedProcess[bytes] | None:
    """Run the fixed security lookup without shell evaluation or diagnostic output."""
    if sys.platform != 'darwin':
        return None
    command = ['/usr/bin/security', 'find-generic-password', '-s', SERVICE, '-a', ACCOUNT]
    if password:
        command.append('-w')
    command.append(str(Path.home() / 'Library/Keychains/login.keychain-db'))
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE if password else subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, env=_environment(), timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def keychain_item_present() -> bool:
    """Flow: Receives external login-Keychain item metadata via security; returns bool for main() without retrieving its password."""
    result = _lookup(False)
    return result is not None and result.returncode == 0


def credential_available() -> bool:
    """Flow: Receives process configuration from queue_unified_turn(); returns bool for live routing without exposing the credential."""
    return _valid_credential(os.environ.get('MODEL_API_KEY', '')) or keychain_item_present()


def _valid_credential(value: str) -> bool:
    """Check that a credential is suitable for an HTTP authorization header."""
    return bool(value) and all(33 <= ord(char) <= 126 for char in value)


def launch_live(command: list[str]) -> int:
    """Flow: Receives an authorized trusted child command list[str] from main(); returns its final exit status int after transient credential injection."""
    if not command or not all(isinstance(arg, str) and arg for arg in command):
        return 2
    result = _lookup(True)
    credential = ''
    if result is not None and result.returncode == 0:
        try:
            credential = result.stdout.decode('ascii').rstrip('\r\n')
        except UnicodeDecodeError:
            pass
    if not _valid_credential(credential):
        credential = os.environ.get('MODEL_API_KEY', '')
    if not _valid_credential(credential):
        print('Meta credential unavailable; no live child was launched.', file=sys.stderr)
        return 1
    environment = _environment()
    environment['MODEL_API_KEY'] = credential
    del credential, result
    try:
        return subprocess.run(command, env=environment, check=False).returncode
    except OSError:
        print('Live child could not be launched.', file=sys.stderr)
        return 1
    finally:
        environment.pop('MODEL_API_KEY', None)


def main(argv: list[str] | None = None) -> int:
    """Flow: Receives external CLI arguments list[str] | None; returns final exit status int from keychain_item_present() or launch_live()."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-keychain', action='store_true', help='Boolean item lookup; never retrieves the password')
    mode.add_argument('--authorize-paid-meta', action='store_true', help='Requires explicit authorization for this live investigation')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Trusted local child command after --; never supply a key')
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.check_keychain:
        if command:
            parser.error('--check-keychain does not launch a command')
        present = keychain_item_present()
        print(f'Keychain item present: {str(present).lower()}')
        return 0 if present else 1
    if not command:
        parser.error('Supply a trusted child command after --')
    return launch_live(command)


if __name__ == '__main__':
    raise SystemExit(main())
