import builtins
import io
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from zymecycle import metadata
from zymecycle.pazy import PazyClient
from zymecycle.ui_adapter import EXAMPLES, run_offline
from zymecycle.ui_findings import display_response, finding_answer, prioritize_limitations
from zymecycle.ui_pazy import discover_pazy_pet, doi_link

APP = Path(__file__).resolve().parents[1] / 'tests/legacy_streamlit_app.py'


@pytest.mark.parametrize('source,expected', [('0.08499999999999999','0.085'), ('0.13','0.13'),
    ('0.0','0'), ('1e-20','1e-20'), ('123456789','1.23457e+8')])
def test_readable_numbers_do_not_change_original(source, expected):
    assert display_response(source) == expected


def test_answer_grounding_and_limitations_preserved():
    view = run_offline(EXAMPLES['Temperature evidence'])['result']
    original = deepcopy(view)
    answer = finding_answer(view)
    assert '85 µmol' in answer and '130 µmol' in answer and 'do not establish' in answer
    assert 'INFERRED conversion; COMPUTED' in answer
    assert [o['original_text'] for o in view['observations']] == ['0.08499999999999999', '0.13']
    assert 'pH 6.5' in answer and 'crystalline powder' in answer
    primary, rest = prioritize_limitations(view)
    assert len(primary) == 3
    assert 'unit declarations are UNKNOWN/UNVERIFIED' in primary[0]
    assert 'direct quantitative' in primary[1]
    assert 'Uncertainty' in primary[2]
    assert set(primary + rest) == set(view['limitations'])
    assert view == original
    view['workflow_status'] = 'PARTIAL'
    assert finding_answer(view).startswith('Partial investigation.')


def test_main_view_rounded_exact_detail_available_and_collapsed_limits():
    at = AppTest.from_file(APP).run()
    at.button(key='investigate').click().run()
    assert not at.exception
    assert at.dataframe[0].value['Recorded response'].tolist() == ['0.085','0.13']
    assert any('0.08499999999999999' in element.value for element in at.json)
    remaining = next(e for e in at.expander if e.label.startswith('All remaining limitations'))
    assert remaining.proto.expanded is False
    text = ' '.join(m.value for m in at.markdown)
    assert all(limit in text for limit in at.session_state['view']['limitations'])


@pytest.fixture
def synthetic():
    return {'id': 876543, 'name': 'SYNTHETIC_PET_ENZYME', 'amino_acid_sequence': 'PRIVATE_SEQUENCE',
            'organism': {'id': 2, 'scientific_name': 'Synthetic organism', 'domain': 'Bacteria', 'phylum': None},
            'substrates': [{'id': 3,'name':'Synthetic PET','abbreviation':'PET','oligomer_only':False}],
            'literature': [{'id': i, 'doi':'10.1234/synthetic', 'title':f'Synthetic paper {i}',
                            'author':'Synthetic author', 'year':2026} for i in range(1,6)],
            'provider_response': 'PRIVATE_PROVIDER', 'embedding': [987.6]}


def client_for(monkeypatch, records, calls):
    def handle(request):
        calls.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={'count':len(records),'results':records,'next':'https://untrusted.invalid/', 'previous':None})
    monkeypatch.setattr('zymecycle.pazy.time.sleep', lambda delay: None)
    return PazyClient(transport=httpx.MockTransport(handle))


def test_bounded_projection_and_explicit_associations(monkeypatch, synthetic):
    calls = []
    result = discover_pazy_pet(1, client=client_for(monkeypatch,[synthetic],calls))
    card = result['result'][0]
    assert calls == [('GET','/api/proteins/',{'substrate':'PET','limit':'1','offset':'0'})]
    assert result['evidence_category'] is None
    assert card['reference_counts'] == {'total':5,'returned':3,'omitted':2,'unresolved_returned':0}
    assert len(card['publications']) == 3
    text = json.dumps(result)
    assert all(secret not in text for secret in ['PRIVATE_', '876543', '987.6', 'amino_acid_sequence'])
    assert 'not marked oligomer-only' in card['pet_associations'][0]
    assert result['status'] == 'PARTIAL'


@pytest.mark.parametrize('refs,total,omitted,unresolved', [(None,None,None,0), ([],0,0,0), ([999],1,0,1)])
def test_unknown_empty_unresolved_references(monkeypatch,synthetic,refs,total,omitted,unresolved):
    synthetic['literature'] = refs
    card = discover_pazy_pet(1,client=client_for(monkeypatch,[synthetic],[]))['result'][0]
    assert card['reference_counts'] == {'total':total,'returned':0 if refs is None else len(refs),
                                       'omitted':omitted,'unresolved_returned':unresolved}
    assert card['publications'] == [] and card['unavailable']


def test_missing_association_is_not_inferred_from_filter(monkeypatch,synthetic):
    synthetic['substrates'] = None
    synthetic['organism'] = None
    synthetic['name'] = None
    card = discover_pazy_pet(1,client=client_for(monkeypatch,[synthetic],[]))['result'][0]
    assert card['pet_associations'] == [] and card['organism'] is None
    assert any('no association inferred' in m for m in card['unavailable'])


@pytest.mark.parametrize('oligomer,phrase', [(True,'oligomer-only annotation'), (None,'status unavailable')])
def test_documented_oligomer_qualifier_preserved(monkeypatch,synthetic,oligomer,phrase):
    synthetic['substrates'][0]['oligomer_only'] = oligomer
    card = discover_pazy_pet(1,client=client_for(monkeypatch,[synthetic],[]))['result'][0]
    assert phrase in card['pet_associations'][0]


def test_unavailable_doi_and_publication_metadata(monkeypatch,synthetic):
    synthetic['literature'] = [{'id':1,'doi':'not-a-doi','title':None,'author':None,'year':None}]
    card = discover_pazy_pet(1,client=client_for(monkeypatch,[synthetic],[]))['result'][0]
    assert card['publications'][0]['doi_url'] is None
    assert any('metadata is unavailable' in m for m in card['unavailable'])
    assert any('no usable DOI' in m for m in card['unavailable'])


@pytest.mark.parametrize('bad', [0,4,True,'3'])
def test_invalid_bounds_never_request(monkeypatch,bad):
    def forbidden(*a,**k): pytest.fail('Invalid limit reached network')
    api = PazyClient(transport=httpx.MockTransport(forbidden))
    assert discover_pazy_pet(bad,client=api)['status'] == 'ERROR'


def test_error_and_no_data(monkeypatch):
    api = PazyClient(transport=httpx.MockTransport(lambda r:httpx.Response(503,text='PRIVATE_ERROR')))
    error = discover_pazy_pet(client=api)
    assert error['status'] == 'ERROR' and 'PRIVATE_ERROR' not in json.dumps(error)
    assert discover_pazy_pet(client=client_for(monkeypatch,[],[]))['status'] == 'NO_DATA'


@pytest.mark.parametrize('value,expected', [(None,None), ('javascript:alert(1)',None),
    ('https://wrong.invalid/paper',None), ('10.1234/a(b)','https://doi.org/10.1234/a%28b%29'),
    ('https://doi.org/10.1234/test','https://doi.org/10.1234/test')])
def test_publication_links_are_doi_only(value,expected):
    assert doi_link(value) == expected


def test_no_storage_logging_or_cross_request_cache(monkeypatch,synthetic,caplog,capsys):
    calls = []
    api = client_for(monkeypatch,[synthetic],calls)
    def forbidden(*a,**k): pytest.fail('PAZy display adapter touched filesystem')
    with monkeypatch.context() as guard:
        for module in (builtins,io,os): guard.setattr(module,'open',forbidden)
        with caplog.at_level(logging.DEBUG):
            first = discover_pazy_pet(1,client=api)
            second = discover_pazy_pet(1,client=api)
    assert len(calls) == 2
    assert first['result'] is not second['result']
    captured = capsys.readouterr()
    assert 'SYNTHETIC_PET_ENZYME' not in caplog.text + captured.out + captured.err
    assert set(vars(api)) == {'_transport','_last_request'}


def test_pazy_ui_transient_no_export_no_paid_reruns(monkeypatch,synthetic):
    calls = []
    monkeypatch.setattr(metadata,'_DEFAULT_CLIENT',client_for(monkeypatch,[synthetic],calls))
    def forbidden(*a,**k): pytest.fail('PAZy discovery invoked Meta/Keychain')
    monkeypatch.setattr('zymecycle.ui_live.subprocess.Popen',forbidden)
    at = AppTest.from_file(APP).run()
    at.sidebar.radio[0].set_value('Live PAZy discovery').run()
    assert calls == []
    at.selectbox[0].set_value(1)
    at.button(key='pazy_search').click().run()
    assert len(calls) == 1 and not at.exception
    text = ' '.join(t.value for t in at.text)
    assert 'SYNTHETIC_PET_ENZYME' in text and 'Synthetic paper 1' in text
    assert 'PRIVATE_' not in text
    assert not at.json and not at.dataframe and not at.get('download_button')
    state = str(at.session_state)
    assert all(s not in state for s in ['SYNTHETIC_', '876543','10.1234/synthetic','PRIVATE_'])
    at.run()
    assert len(calls) == 1 and not at.text
    at.sidebar.radio[0].set_value('Offline demonstration').run()
    assert 'view' not in at.session_state and len(calls)==1
