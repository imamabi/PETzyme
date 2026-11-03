import json

import httpx

from petzyme import metadata
from petzyme.pazy import PazyClient
from scripts.demo_build01 import main, nrel_examples, pazy_smoke_summary


def test_local_demonstrations(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow\nE1,ACDEF,0\nE2,ACDEG,2\n')
    examples = nrel_examples(str(path))
    assert examples['scenario_a']['result']['recorded_zeros'] == 1
    assert examples['scenario_b']['status'] == 'NO_DATA'
    assert nrel_examples(str(tmp_path / 'missing'))['local_error']['status'] == 'ERROR'


def test_live_demo_never_emits_records(monkeypatch):
    record = {'id': 7, 'name': 'DO_NOT_EMIT_METADATA', 'amino_acid_sequence': 'ACDEF',
              'literature': [{'id': 9, 'doi': 'DO_NOT_EMIT_DOI', 'author': 'DO_NOT_EMIT_AUTHOR',
                              'title': 'DO_NOT_EMIT_TITLE', 'year': 2020}]}
    def handler(request):
        if request.url.path == '/api/proteins/':
            return httpx.Response(200, json={'count': 1, 'next': None, 'previous': None, 'results': [record]})
        return httpx.Response(200, json=record)
    monkeypatch.setattr(metadata, '_DEFAULT_CLIENT', PazyClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr('petzyme.pazy.time.sleep', lambda delay: None)
    summary = pazy_smoke_summary()
    assert summary['metadata_status'] == 'SUCCESS'
    assert summary['returned_reference_count'] == summary['embedded_reference_count'] == 1
    encoded = json.dumps(summary, allow_nan=False)
    assert 'DO_NOT_EMIT' not in encoded and 'ACDEF' not in encoded


def test_service_failure_is_reported(monkeypatch):
    monkeypatch.setattr(metadata, '_DEFAULT_CLIENT', PazyClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(503, text='DO_NOT_EMIT'))))
    assert pazy_smoke_summary()['search_status'] == 'ERROR'


def test_cli_offline_default(tmp_path, capsys):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow\nE1,ACDEF,1\n')
    assert main(['--csv', str(path)]) == 0
    assert 'pazy_smoke' not in capsys.readouterr().out
    assert main(['--csv', str(tmp_path / 'missing')]) == 1
