"""Mocked asset setup/integrity tests; synthetic bytes are never used as real embeddings."""
import hashlib
import json
import httpx
import pytest
from scripts import prepare_esm2 as setup


@pytest.fixture
def assets(tmp_path,monkeypatch):
    manifest=json.loads((setup.ROOT/'MODEL_PROVENANCE.json').read_text())
    manifest['files']={}
    (tmp_path/'MODEL_PROVENANCE.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    data={name:('SYNTHETIC_ASSET_'+name).encode() for name in setup.FILES}
    siblings=[]
    for name,content in data.items():
        entry={'rfilename':name,'size':len(content),
               'blobId':hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()}
        if name=='model.safetensors': entry['lfs']={'sha256':hashlib.sha256(content).hexdigest()}
        siblings.append(entry)
    metadata={'sha':manifest['revision'],'cardData':{'license':'mit'},'siblings':siblings}
    calls=[]
    def handler(request):
        calls.append(request.url)
        if request.url.path.startswith('/api/models/'):
            return httpx.Response(200,json=metadata)
        return httpx.Response(200,content=data[request.url.path.rsplit('/',1)[-1]])
    original=httpx.Client
    monkeypatch.setattr(setup.httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    return tmp_path,metadata,data,calls


def test_only_five_pinned_assets_downloaded_and_verified(assets):
    root,metadata,data,calls=assets
    result=setup.prepare_checkpoint()
    assert result['files_verified']==5 and len(calls)==6
    assert all(url.scheme=='https' and url.host=='huggingface.co' for url in calls)
    assert {url.path.rsplit('/',1)[-1] for url in calls[1:]}==set(setup.FILES)
    manifest=json.loads((root/'MODEL_PROVENANCE.json').read_text())
    assert manifest['downloaded_at_utc'] and set(manifest['files'])==set(setup.FILES)
    for name,entry in manifest['files'].items():
        assert entry['sha256']==hashlib.sha256(data[name]).hexdigest()
    before = (root/'MODEL_PROVENANCE.json').read_bytes()
    setup.prepare_checkpoint()
    assert (root/'MODEL_PROVENANCE.json').read_bytes() == before


@pytest.mark.parametrize('kind',['revision','license','hash'])
def test_unverified_assets_fail_without_recording_success(assets,kind):
    root,metadata,data,calls=assets
    if kind=='revision': metadata['sha']='wrong'
    if kind=='license': metadata['cardData']['license']='unknown'
    if kind=='hash': next(s for s in metadata['siblings'] if s['rfilename']=='model.safetensors')['lfs']['sha256']='0'*64
    with pytest.raises(ValueError): setup.prepare_checkpoint()
    assert not json.loads((root/'MODEL_PROVENANCE.json').read_text())['files']
