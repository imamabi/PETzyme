import builtins
import io
import json
import logging
import os
from copy import deepcopy

import httpx
import pytest

from zymecycle import metadata
from zymecycle.metadata import get_associated_literature, get_pazy_metadata, search_pazy_pet
from zymecycle.pazy import PazyClient


@pytest.fixture
def record():
    return {'id': 7, 'name': 'SYNTHETIC_ENZYME', 'amino_acid_sequence': 'ACDEF',
            'uniprot_accession': 'SYNTHETIC',
            'organism': {'id': 2, 'scientific_name': 'Synthetic organism', 'domain': 'Bacteria', 'phylum': None},
            'substrates': [{'id': 3, 'name': 'Synthetic PET', 'abbreviation': 'PET', 'oligomer_only': True}],
            'literature': [{'id': 9, 'doi': 'synthetic-doi', 'author': 'Synthetic Author',
                            'title': 'Synthetic Title', 'year': 2020}]}


def make_client(monkeypatch, handler):
    monkeypatch.setattr('zymecycle.pazy.time.sleep', lambda seconds: None)
    return PazyClient(transport=httpx.MockTransport(handler))


def test_metadata_and_explicit_literature(record, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=record)
    result = get_pazy_metadata(7, client=make_client(monkeypatch, handler))
    assert result['status'] == 'SUCCESS'
    assert result['evidence_category'] is None  # Metadata is not experimental evidence.
    protein = result['result']
    assert protein['id'] == 7 and protein['amino_acid_sequence'] == 'ACDEF'
    assert protein['substrates'][0]['oligomer_only'] is True
    references = get_associated_literature(protein)
    assert references['result'][0]['record']['doi'] == 'synthetic-doi'
    assert references['result'][0]['resolution'] == 'EMBEDDED'
    assert references['provenance'] == result['provenance']
    assert len(calls) == 1 and calls[0].url.path == '/api/proteins/7/'
    json.dumps(result, allow_nan=False)
    json.dumps(references, allow_nan=False)


def test_search_construction(record, monkeypatch):
    def handler(request):
        assert request.url.path == '/api/proteins/'
        assert dict(request.url.params) == {'substrate': 'PET', 'limit': '1', 'offset': '0'}
        return httpx.Response(200, json={'count': 1, 'next': None, 'previous': None, 'results': [record]})
    result = search_pazy_pet(1, client=make_client(monkeypatch, handler))
    assert result['result'][0]['id'] == 7
    assert result['result'][0]['source']['name'] == 'PAZy'
    assert result['provenance'][0]['retrieved_at_utc']


@pytest.mark.parametrize('status', [404, 429, 503])
def test_request_failure(record, monkeypatch, status):
    api = make_client(monkeypatch, lambda request: httpx.Response(status, text='PRIVATE_RECORD'))
    result = get_pazy_metadata(7, client=api)
    assert result['status'] == 'ERROR' and result['result'] is None
    assert 'PRIVATE_RECORD' not in json.dumps(result)


def test_timeout_does_not_break_local_analysis(monkeypatch, tmp_path):
    from zymecycle.observations import load_nrel, parse_nrel
    def handler(request):
        raise httpx.ReadTimeout('PRIVATE_RECORD', request=request)
    assert search_pazy_pet(client=make_client(monkeypatch, handler))['status'] == 'ERROR'
    path = tmp_path / 'local.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow\nE1,ACDEF,1\n')
    assert parse_nrel(load_nrel(str(path))['result'])['status'] == 'SUCCESS'


@pytest.mark.parametrize('change', [
    {'id': True}, {'name': 123}, {'amino_acid_sequence': []}, {'literature': 'invalid'},
    {'organism': []}, {'substrates': 'PET'}, {'substrates': [{'id': 1, 'oligomer_only': 'false'}]},
])
def test_malformed_metadata(record, monkeypatch, change):
    record.update(change)
    result = get_pazy_metadata(7, client=make_client(monkeypatch, lambda r: httpx.Response(200, json=record)))
    assert result['status'] == 'ERROR'


@pytest.mark.parametrize('references', [[9], [{'id': 9}], [{'id': 9, 'doi': [], 'title': 'bad'}], [None]])
def test_unresolved_references_no_invented_relationships(record, monkeypatch, references):
    record['literature'] = references
    api = make_client(monkeypatch, lambda r: httpx.Response(200, json=record))
    protein = get_pazy_metadata(7, client=api)['result']
    result = get_associated_literature(protein)
    assert result['status'] == 'PARTIAL'
    assert all(item['record'] is None and item['resolution'] == 'UNRESOLVED' for item in result['result'])


def test_missing_empty_and_bounded_literature(record, monkeypatch):
    api = make_client(monkeypatch, lambda r: httpx.Response(200, json=record))
    for value, status in [(None, 'PARTIAL'), ([], 'NO_DATA')]:
        record['literature'] = value
        assert get_associated_literature(get_pazy_metadata(7, client=api)['result'])['status'] == status
    record['literature'] = [{'id': i, 'doi': None, 'author': None, 'title': 'Synthetic', 'year': None}
                            for i in range(1, 9)]
    protein = get_pazy_metadata(7, client=api)['result']
    assert len(get_associated_literature(protein)['result']) == 5
    assert get_associated_literature(protein)['status'] == 'PARTIAL'
    assert get_associated_literature(protein, limit=6)['status'] == 'ERROR'


def test_no_file_access_logging_or_cache(record, monkeypatch, caplog, capsys, tmp_path):
    before = deepcopy(record)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=record)
    api = make_client(monkeypatch, handler)
    def forbidden(*args, **kwargs):
        pytest.fail('Scientific PAZy adapter attempted filesystem access')
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as guard:
        guard.setattr(builtins, 'open', forbidden)
        guard.setattr(io, 'open', forbidden)
        guard.setattr(os, 'open', forbidden)
        with caplog.at_level(logging.DEBUG):
            for _ in range(2):
                protein = get_pazy_metadata(7, client=api)['result']
                get_associated_literature(protein)
    assert len(calls) == 2
    assert record == before
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert 'SYNTHETIC_ENZYME' not in caplog.text + captured.out + captured.err
    assert 'ACDEF' not in caplog.text + captured.out + captured.err
    assert set(vars(api)) == {'_transport', '_last_request'}


def test_no_external_url_dereference_and_field_projection(record, monkeypatch):
    record['malicious_url'] = 'https://panda.invalid/records/'
    record['literature'][0]['url'] = 'https://panda.invalid/'
    api = make_client(monkeypatch, lambda r: httpx.Response(200, json=record))
    protein = get_pazy_metadata(7, client=api)['result']
    assert 'panda' not in json.dumps(protein)
    assert get_associated_literature(protein)['status'] == 'SUCCESS'


@pytest.mark.parametrize('limit', [0, -1, 4, True, '1'])
def test_adapter_search_bound(limit, monkeypatch):
    api = make_client(monkeypatch, lambda request: pytest.fail('Invalid adapter limit reached HTTP'))
    assert search_pazy_pet(limit, client=api)['status'] == 'ERROR'


def test_structured_reference_counts_are_complete(record, monkeypatch):
    api = make_client(monkeypatch, lambda r: httpx.Response(200, json=record))
    record['literature'] = [{'id': i, 'doi': None, 'author': None, 'title': 'Synthetic', 'year': None}
                            for i in range(1, 9)]
    protein = get_pazy_metadata(7, client=api)['result']
    result = get_associated_literature(protein)
    assert result['reference_counts'] == {'total': 8, 'returned': 5, 'omitted': 3, 'unresolved_returned': 0}
    assert result['status'] == 'PARTIAL'
    protein['literature'] = None
    assert get_associated_literature(protein)['reference_counts'] == {
        'total': None, 'returned': 0, 'omitted': None, 'unresolved_returned': 0}
    protein['literature'] = []
    assert get_associated_literature(protein)['reference_counts']['omitted'] == 0
    protein['literature'] = [{'reference_id': 1, 'resolution': 'UNRESOLVED', 'record': None}]
    assert get_associated_literature(protein)['reference_counts']['unresolved_returned'] == 1
