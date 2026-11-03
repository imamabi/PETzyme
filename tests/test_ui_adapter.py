import json
import pytest
from test_agent import csv_path
from test_esm import SyntheticRuntime
from zymecycle import esm
from zymecycle.results import outcome
from zymecycle.ui_adapter import EXAMPLES, run_offline, investigate_with_backend, present_investigation
from scripts.demo_build03 import ScenarioMeta, QUESTIONS
from zymecycle.meta import MetaModelClient
import httpx


@pytest.mark.parametrize('example,tools,attempts', [
    ('Temperature evidence', 3, 4), ('Missing conditions', 4, 5),
    ('ESM-2 similarity', 2, 3), ('Next experiment', 4, 5)])
def test_four_workflows_real_tools_with_deterministic_orchestration(csv_path, monkeypatch, example, tools, attempts):
    monkeypatch.setattr(esm, 'load_esm2_runtime', lambda: SyntheticRuntime())
    updates = []
    result = run_offline(EXAMPLES[example], csv_path=csv_path, on_update=updates.append)
    assert result['status'] == 'SUCCESS'
    view = result['result']
    assert view['workflow_status'] == 'COMPLETE' and view['mode'] == 'DETERMINISTIC_OFFLINE'
    assert view['model_requests'] == attempts and view['tool_attempts'] == tools
    assert len(updates) == tools and view['limitations']
    text = json.dumps(view, allow_nan=False)
    assert 'ACDEF'*20 not in text and '"rows"' not in text and '"vector"' not in text
    if example == 'Temperature evidence':
        assert [o['original_text'] for o in view['observations']] == ['0.085', '0.13']
        assert not view['comparison']['direct_comparison_supported']
    if example == 'Missing conditions':
        assert view['observations'] == [] and len(view['gaps']) == 3
        assert view['record_identity']['enzyme_identifier'] == 'ESM050'
        assert {c['condition']['pet_form'] for c in view['conditions']} == {'CRYSTALLINE_POWDER','AMORPHOUS_FILM'}
    if example == 'ESM-2 similarity':
        assert view['similarity']['coverage']['searched_candidates'] == 1
        assert 'unknown' in view['headline']
    if example == 'Next experiment':
        seen = set()
        for c in view['plan']['recommendations']:
            assert c['evidence_category'] == 'HYPOTHETICAL'
            assert set(c['prerequisites']) <= seen
            seen.add(c['action_id'])


def test_real_nrel_values_are_not_reconstructed_from_facts():
    view = run_offline(EXAMPLES['Temperature evidence'])['result']
    assert [o['original_text'] for o in view['observations']] == ['0.08499999999999999', '0.13']
    assert all(o['source_row'] == 160 for o in view['observations'])
    assert all('column=activity_at_6.5_' in o['source_locator'] for o in view['observations'])


def test_unsupported_offline_question_does_not_execute(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('No execution for unsupported question')
    monkeypatch.setattr('zymecycle.ui_adapter.run_investigation', forbidden)
    assert run_offline('Invent a catalytic prediction')['status'] == 'ERROR'


def test_errors_and_partial_views_preserve_actual_status(csv_path):
    result = run_offline(EXAMPLES['Temperature evidence'], csv_path='nonexistent-source.csv')
    assert result['status'] == 'PARTIAL' and result['result']['workflow_status'] == 'PARTIAL'
    assert result['result']['observations'] == [] and result['result']['issues']
    assert result['result']['progress'][0]['status'] == 'ERROR'


def test_pazy_records_do_not_enter_view_or_session_data(monkeypatch, csv_path):
    from zymecycle import metadata
    from zymecycle.pazy import PazyClient
    record = {'id': 777, 'name': 'PRIVATE_NAME', 'amino_acid_sequence': 'ACDEF'*20, 'literature': []}
    def handler(request):
        return httpx.Response(200,json=({'count':1,'results':[record],'next':None} if request.url.path=='/api/proteins/' else record))
    monkeypatch.setattr(metadata,'_DEFAULT_CLIENT',PazyClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr('zymecycle.pazy.time.sleep',lambda delay:None)
    result = investigate_with_backend(QUESTIONS['D'], MetaModelClient(transport=httpx.MockTransport(ScenarioMeta('D').respond)), csv_path=csv_path)
    text = json.dumps(result)
    assert 'PRIVATE_NAME' not in text and '777' not in text and 'ACDEF'*20 not in text
    assert result['result']['question'] == 'PAZy investigation — record details withheld'
    assert 'withheld' in text
    assert result['result']['restricted_annotation_status']
    assert all('PRIVATE_NAME' not in item and '777' not in item
               for item in result['result']['restricted_annotation_status'])


def test_evidence_listing_headline_counts_observations_not_fallback():
    report = {
        "question": "List recorded ESM050 observations",
        "persistence": "NREL_ONLY",
        "provider_mode": "LIVE",
        "status": "COMPLETE",
        "model_requests": 1,
        "tool_attempts": 4,
        "limitations": [],
        "facts": [],
        "validation": {
            "issues": [],
            "scientific_scope": {"request": {"purpose": "DIRECT"}},
        },
    }
    updates = [{
        "call_id": "c1", "name": "assess_nrel_evidence", "status": "SUCCESS",
        "summary": "observations returned", "model_requests": 1, "tool_attempts": 1,
        "data": {"observations": [{"original_text": "0.13"}, {"original_text": "0.085"}]},
        "limitations": [], "provenance": [],
    }]
    view = present_investigation(outcome("SUCCESS", report, [], [], None), updates)["result"]
    assert "2 source-bound NREL observations" in view["headline"]
    assert "No publishable structured evidence" not in view["headline"]
