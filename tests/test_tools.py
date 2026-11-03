import builtins
import io
import inspect
import json
import os
from pathlib import Path
from typing import get_type_hints

import httpx
import pytest

from zymecycle import metadata
from zymecycle.pazy import PazyClient
from zymecycle.tools import TOOL_FUNCTIONS, describe_tools, invoke_tool


def test_discovery_has_typed_callable_contracts():
    catalog = describe_tools()
    assert catalog['status'] == 'SUCCESS' and len(catalog['result']) == 19
    assert 'discover_pet_enzymes' in {item['name'] for item in catalog['result']}
    assert json.loads(json.dumps(catalog, allow_nan=False)) == catalog
    for descriptor in catalog['result']:
        function = TOOL_FUNCTIONS[descriptor['name']]
        hints = get_type_hints(function)
        assert hints['return']
        assert all(name in hints for name in inspect.signature(function).parameters)
        assert 'Flow: Receives' in inspect.getdoc(function)
        assert descriptor['input_schema']['additionalProperties'] is False
        assert 'client' not in descriptor['input_schema']['properties']


def test_composable_json_workflow(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow\nE1,ACDEF,1.00\nE2,ACDEG,0\n')
    table = invoke_tool('load_nrel', {'csv_path': str(path)})
    parsed = invoke_tool('parse_nrel', {'table': table['result']})
    summary = invoke_tool('summarize_condition', {
        'observations': parsed['result']['observations'], 'temperature_c': 40,
        'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'})
    assert summary['status'] == 'SUCCESS'
    assert summary['result']['recorded_zeros'] == 1
    assert invoke_tool('select_nrel_protein', {'table': table['result'], 'enzyme_identifier': 'E1'})['status'] == 'SUCCESS'
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize('name,arguments', [
    ('unknown', {}), ('get_pazy_metadata', {'protein_id': True}),
    ('get_pazy_metadata', {'protein_id': 1, 'client': {}}),
    ('parse_nrel', {'table': {}}), ('filter_observations', {'observations': 'invalid'}),
    ('summarize_condition', {'observations': [], 'temperature_c': float('nan'),
                             'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}),
    ('summarize_condition', {'observations': [], 'temperature_c': 40,
                             'ph': 6.5, 'pet_form': 'invented'}),
])
def test_bad_json_arguments(name, arguments):
    assert invoke_tool(name, arguments)['status'] == 'ERROR'


def test_dispatch_pazy_is_ephemeral(monkeypatch, tmp_path, caplog, capsys):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'id': 1, 'name': 'SYNTHETIC_SECRET',
                                         'amino_acid_sequence': 'ACDEF', 'literature': []})
    monkeypatch.setattr(metadata, '_DEFAULT_CLIENT', PazyClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr('zymecycle.pazy.time.sleep', lambda delay: None)
    monkeypatch.chdir(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Dispatcher attempted to persist a PAZy result')
    with monkeypatch.context() as guard:
        guard.setattr(builtins, 'open', forbidden)
        guard.setattr(io, 'open', forbidden)
        guard.setattr(os, 'open', forbidden)
        for _ in range(2):
            result = invoke_tool('get_pazy_metadata', {'protein_id': 1})
            assert result['status'] == 'SUCCESS'
            assert invoke_tool('get_associated_literature', {'protein': result['result']})['status'] == 'NO_DATA'
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert 'SYNTHETIC_SECRET' not in caplog.text + captured.out + captured.err
    assert all(request.url.host == 'api.pazy.eu' for request in calls)
    descriptor = next(item for item in describe_tools()['result'] if item['name'] == 'get_pazy_metadata')
    assert descriptor['output_persistence'] == 'FORBIDDEN_PAZY_DERIVED'


def test_identity_composition_and_independent_sources(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow\nE1,acdef,1\n')
    table = invoke_tool('load_nrel', {'csv_path': str(path)})['result']
    nrel_protein = invoke_tool('select_nrel_protein', {'table': table, 'enzyme_identifier': 'E1'})['result']
    api = PazyClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        'id': 1, 'name': 'different name', 'amino_acid_sequence': 'ACDEF', 'literature': []})))
    monkeypatch.setattr(metadata, '_DEFAULT_CLIENT', api)
    pazy_protein = invoke_tool('get_pazy_metadata', {'protein_id': 1})['result']
    result = invoke_tool('compare_identity', {'nrel_protein': nrel_protein, 'pazy_protein': pazy_protein})
    assert result['status'] == 'SUCCESS'
    assert result['result']['identity'] == 'EXACT_SEQUENCE_MATCH'
    assert [source['name'] for source in result['provenance']] == ['NREL', 'PAZy']
    assert result['result']['nrel_identifier'] == 'E1'
    assert result['result']['pazy_identifier'] == 1


def test_huge_condition_and_invalid_name_are_structured_errors():
    assert invoke_tool('summarize_condition', {'observations': [], 'temperature_c': 10 ** 400,
                                               'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'})['status'] == 'ERROR'
    assert invoke_tool([], {})['status'] == 'ERROR'


def test_public_argument_validator():
    from zymecycle.tools import validate_arguments
    schema = {'type': 'object', 'properties': {'limit': {'type': 'integer'}},
              'required': ['limit'], 'additionalProperties': False}
    assert validate_arguments({'limit': 1}, schema)
    assert not validate_arguments({'limit': True}, schema)
    assert not validate_arguments({'limit': 1, 'extra': 2}, schema)
