"""Explicit download of five pinned official checkpoint/tokenizer assets, never a dataset."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
FILES = ('config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt')


def prepare_checkpoint() -> dict:
    """Flow: Receives external explicit setup invocation and MODEL_PROVENANCE.json; returns final verified asset metadata dict for main()."""
    manifest_path = ROOT / 'MODEL_PROVENANCE.json'
    manifest = json.loads(manifest_path.read_text())
    checkpoint, revision = manifest['checkpoint'], manifest['revision']
    if checkpoint != 'facebook/esm2_t12_35M_UR50D' or len(revision) != 40:
        raise ValueError('Unexpected checkpoint specification')
    directory = ROOT / manifest['local_directory']
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(60, connect=10), trust_env=False, follow_redirects=True) as client:
        response = client.get(f'https://huggingface.co/api/models/{checkpoint}/revision/{revision}', params={'blobs': 'true'})
        response.raise_for_status()
        metadata = response.json()
        if metadata['sha'] != revision or metadata['cardData']['license'] != 'mit':
            raise ValueError('Official checkpoint revision/license mismatch')
        entries = {item['rfilename']: item for item in metadata['siblings']}
        verified = {}
        for name in FILES:
            entry = entries[name]
            size = entry['size']
            if not 0 < size <= (150_000_000 if name == 'model.safetensors' else 100_000):
                raise ValueError('Unexpected asset size')
            target = directory / name
            partial = target.with_suffix(target.suffix + '.partial')
            if not target.exists():
                try:
                    with client.stream('GET', f'https://huggingface.co/{checkpoint}/resolve/{revision}/{name}') as download:
                        download.raise_for_status()
                        if download.url.scheme != 'https':
                            raise ValueError('HTTPS required')
                        count = 0
                        with partial.open('xb') as handle:
                            for chunk in download.iter_bytes(65536):
                                count += len(chunk)
                                if count > size:
                                    raise ValueError('Asset exceeded published size')
                                handle.write(chunk)
                    partial.replace(target)
                finally:
                    partial.unlink(missing_ok=True)
            content = target.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            valid = (digest == entry['lfs']['sha256'] if 'lfs' in entry else
                     hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == entry['blobId'])
            if len(content) != size or not valid:
                raise ValueError('Asset failed official integrity check: ' + name)
            verified[name] = {'bytes': size, 'sha256': digest, 'published_git_blob': entry['blobId'],
                              'published_lfs_sha256': entry.get('lfs', {}).get('sha256')}
    pinned = manifest.get('files')
    if pinned:
        if any(name not in pinned or pinned[name].get('sha256') != entry['sha256']
               or pinned[name].get('bytes') != entry['bytes'] for name, entry in verified.items()):
            raise ValueError('Official assets do not match the pinned local manifest')
    else:
        manifest['files'] = verified
        manifest['downloaded_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return {'revision': revision, 'files_verified': len(verified), 'bytes': sum(f['bytes'] for f in verified.values())}


def main() -> None:
    """Flow: Receives external CLI invocation; prints final prepare_checkpoint() verification counts without scientific records or secrets."""
    print(json.dumps(prepare_checkpoint(), indent=2))


if __name__ == '__main__':
    main()
