"""Acquire the one authorized PAZy activity archive without persisting API records."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import httpx

from zymecycle.activity_snapshot import (
    FILENAME,
    MAX_COMPRESSED,
    SHA256,
    URL,
    load_pazy_activity_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


class PreparationResult(TypedDict):
    checksum: str
    schema: str
    reused: bool


def _verify(path: Path) -> None:
    """Verify exact source bytes and the existing archive schema without reporting records."""
    if path.is_symlink() or path.stat().st_size > MAX_COMPRESSED:
        raise ValueError('Unsafe or oversized PAZy archive')
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('PAZy archive checksum mismatch')
    result = load_pazy_activity_snapshot(str(path))
    if result['status'] != 'SUCCESS':
        raise ValueError('PAZy archive schema or counts failed validation')


def prepare_snapshot(root: Path) -> PreparationResult:
    """Flow: Receives an external local root Path; returns PreparationResult for main() after validating the pinned activity archive."""
    directory = root / 'data' / 'pazy'
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('PAZy data directory must stay inside the project')
    target = directory / FILENAME
    if target.is_symlink():
        raise ValueError('Refusing a symlinked PAZy archive')
    if target.exists():
        _verify(target)
        return {'checksum': 'verified', 'schema': 'verified', 'reused': True}
    partial = directory / (FILENAME + '.part')
    created = False
    try:
        with httpx.Client(timeout=httpx.Timeout(60, connect=15), follow_redirects=True, trust_env=False) as client:
            with client.stream('GET', URL) as response:
                response.raise_for_status()
                if any(item.url.scheme != 'https' for item in (*response.history, response)):
                    raise ValueError('PAZy archive download must use HTTPS')
                total = 0
                with partial.open('xb') as output:
                    created = True
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > MAX_COMPRESSED:
                            raise ValueError('PAZy archive size limit exceeded')
                        output.write(chunk)
        _verify(partial)
        partial.replace(target)
    finally:
        if created:
            partial.unlink(missing_ok=True)
    return {'checksum': 'verified', 'schema': 'verified', 'reused': False}


def main() -> None:
    """Flow: Receives external setup invocation; prints aggregate PreparationResult from prepare_snapshot() without source records."""
    print(prepare_snapshot(ROOT))


if __name__ == '__main__':
    main()
