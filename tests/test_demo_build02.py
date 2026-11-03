import builtins
import io
import json
import os

import httpx
from scripts.demo_build02 import bounded_identity_check, local_demonstrations, main, privacy_summary
from zymecycle.observations import load_nrel
from zymecycle.pazy import PazyClient


def table(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow,activity_at_6.5_60_cryPow,activity_at_6.5_40_aFilm\n'
                    'ESM050,ACDEF,0,2,3\nLCC-ICCG,ACDEG,4,,5\n')
    return load_nrel(str(path))['result']


def test_shared_sequence_identity_and_omitted_references(tmp_path, monkeypatch, caplog, capsys):
    local = table(tmp_path)
    requests = []
    record = {'id': 7, 'name': 'PRIVATE_PAZY_NAME', 'amino_acid_sequence': 'acdef',
              'literature': [{'id': i, 'doi': 'PRIVATE_PAZY_DOI', 'author': 'PRIVATE_AUTHOR',
                              'title': 'PRIVATE_TITLE', 'year': 2020} for i in range(1, 9)]}
    def handler(request):
        requests.append(request)
        if request.url.path == '/api/proteins/':
            assert request.url.params['limit'] == '3'
            return httpx.Response(200, json={'count': 1, 'next': None, 'previous': None, 'results': [record]})
        assert request.url.path == '/api/proteins/7/'
        return httpx.Response(200, json=record)
    client = PazyClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr('zymecycle.pazy.time.sleep', lambda delay: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('PAZy identity check attempted file access')
    with monkeypatch.context() as guard:
        guard.setattr(builtins, 'open', forbidden)
        guard.setattr(io, 'open', forbidden)
        guard.setattr(os, 'open', forbidden)
        check = bounded_identity_check(local, client=client)
        summary = privacy_summary(check)
    assert check['result']['identity']['identity'] == 'EXACT_SEQUENCE_MATCH'
    assert check['result']['identity']['nrel_identifier'] == 'ESM050'
    assert check['result']['identity']['pazy_identifier'] == 7
    assert [s['name'] for s in check['provenance']] == ['NREL', 'PAZy']
    assert check['result']['reference_counts']['omitted'] == 3
    assert len(requests) == 2
    encoded = json.dumps(summary)
    assert 'PRIVATE_' not in encoded and 'acdef' not in encoded and 'sha256' not in encoded
    assert 'pazy_identifier' not in encoded
    captured = capsys.readouterr()
    assert 'PRIVATE_' not in captured.out + captured.err + caplog.text


def test_name_match_does_not_establish_sequence_identity(tmp_path):
    local = table(tmp_path)
    record = {'id': 7, 'name': 'ESM050', 'amino_acid_sequence': 'WWWW'}
    api = PazyClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        'count': 1, 'next': None, 'previous': None, 'results': [record]})))
    result = bounded_identity_check(local, client=api)
    assert result['status'] == 'NO_DATA'
    assert result['result']['identity'] is None
    assert [source['name'] for source in result['provenance']] == ['NREL', 'PAZy']


def test_service_failure_and_changed_detail(tmp_path, monkeypatch):
    local = table(tmp_path)
    api = PazyClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert bounded_identity_check(local, client=api)['status'] == 'ERROR'
    def handler(request):
        if request.url.path == '/api/proteins/':
            return httpx.Response(200, json={'count': 1, 'next': None, 'previous': None,
                                            'results': [{'id': 1, 'amino_acid_sequence': 'ACDEF'}]})
        return httpx.Response(200, json={'id': 1, 'amino_acid_sequence': 'ACDEG'})
    monkeypatch.setattr('zymecycle.pazy.time.sleep', lambda delay: None)
    api = PazyClient(transport=httpx.MockTransport(handler))
    assert bounded_identity_check(local, client=api)['status'] == 'NO_DATA'


def test_local_demonstrations_and_cli(tmp_path, capsys):
    local = table(tmp_path)
    results = local_demonstrations(local)
    assert results['within_condition']['result']['descriptive_within_column_supported']
    assert results['different_pet_forms']['result']['comparability']['category'] == 'NOT_COMPARABLE'
    assert results['missing_condition']['status'] == 'PARTIAL'
    assert main(['--csv', local['source']['locator']]) == 0
    assert 'live_identity' not in capsys.readouterr().out
    assert main(['--csv', str(tmp_path / 'missing')]) == 1
