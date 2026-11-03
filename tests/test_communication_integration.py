"""Mock provider + real tools; every generated explanation here is synthetic, not Muse acceptance."""
from copy import deepcopy
import json
import os
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from zymecycle import agent, esm, ui_adapter
from zymecycle.agent_evidence import model_evidence
from zymecycle.agent_reporting import format_research_report
from zymecycle.meta import MetaModelClient, MODEL
from zymecycle.ui_adapter import EXAMPLES, OfflineWorkflow, investigate_with_backend
from test_communication import draft_from_facts
from test_esm import SyntheticRuntime

APP = Path(__file__).resolve().parents[1] / 'tests/legacy_streamlit_app.py'


def expanded_facts(messages):
    sources, facts = {}, {}
    for message in messages:
        if message['role'] != 'tool': continue
        reply = json.loads(message['content']); sources.update(reply.get('source_catalog', {}))
        for f in reply['facts']:
            if 'same_as' in f:
                item = {**facts[f['same_as']], 'id':f['id']}
            else:
                item = {k:v for k,v in f.items() if k != 'source_refs'}
                item['sources'] = f.get('sources', [sources[r] for r in f.get('source_refs',[])])
            facts[f['id']] = item
    return list(facts.values())


class CommunicationScenario:
    def __init__(self, example='Integrated experiment advisor', mode='valid'):
        self.question, self.mode = EXAMPLES[example], mode
        self.decisions = OfflineWorkflow(self.question)
        self.requests, self.sizes = [], []

    def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload); self.sizes.append(len(request.content))
        message = self.decisions.complete(payload['messages'], payload['tools'])
        finish = 'tool_calls' if message.get('tool_calls') else 'stop'
        if finish == 'stop':
            content = json.loads(message['content'])
            draft = draft_from_facts(expanded_facts(payload['messages']))
            if self.mode == 'claim': draft['evidence'][0]['text'] = 'Temperature causes superior PET activity at 60°C.'
            if self.mode == 'citation': draft['evidence'][0].pop('fact_id')
            if self.mode == 'omitted_fact': content['fact_ids'] = content['fact_ids'][:-1]
            if self.mode == 'legacy_extra': content['unsupported_conclusion'] = 'Superior activity'
            if self.mode == 'length': finish = 'length'
            if self.mode != 'absent': content['explanation'] = draft
            message['content'] = json.dumps(content)
        return httpx.Response(200, json={'model':MODEL, 'choices':[{'finish_reason':finish, 'message':message}]})


def backend(scenario):
    return MetaModelClient(transport=httpx.MockTransport(scenario.respond))


@pytest.mark.parametrize('example', ['Integrated experiment advisor', 'Missing conditions', 'ESM-2 similarity'])
def test_actual_contract_with_real_local_scientific_execution(example, monkeypatch):
    monkeypatch.setattr(esm, 'load_esm2_runtime', lambda: SyntheticRuntime())
    scenario = CommunicationScenario(example)
    result = agent.run_investigation(scenario.question, backend(scenario))
    report = result['result']
    assert report['communication']['status'] == 'VERIFIED_EXTRACTIVE'
    assert report['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
    assert report['validation']['selection_counts']['omitted'] == 0
    assert report['model_requests'] == len(scenario.requests) <= 6 and report['tool_attempts'] <= 8
    if example == 'Integrated experiment advisor':
        assert report['status'] == 'PARTIAL' and report['tool_attempts'] == 1 and len(scenario.requests) == 2
        assert report['validation']['selection_counts'] == {'required':23,'selected':23,'omitted':0}
        assert all(f['ephemeral'] for f in report['facts'])
        assert 'PAZy publication-derived assay:' not in format_research_report(report)
    if example == 'ESM-2 similarity':
        statements = report['communication']['sections']['evidence']
        assert all(s['evidence_category'] == 'COMPUTED' for s in statements)
        assert any('not measured activity' in s['text'] for s in statements)
        assert 'UNKNOWN' in ' '.join(s['text'] for s in report['communication']['sections']['uncertainty'])
    if example == 'Missing conditions':
        assert not any(s['evidence_category']=='MEASURED' for s in report['communication']['sections']['evidence'])
        assert any('INSUFFICIENT_INFORMATION' in s['text'] for s in report['communication']['sections']['evidence'])
        assert any('missing is not zero' in s['text'] for s in report['communication']['sections']['uncertainty'])


@pytest.mark.parametrize('mode', ['claim','citation','absent','omitted_fact','legacy_extra','length'])
def test_failures_fall_back_without_extra_requests_or_grounding_relaxation(mode):
    scenario = CommunicationScenario(mode=mode)
    report = agent.run_investigation(scenario.question, backend(scenario))['result']
    assert report['communication']['status'] == 'DETERMINISTIC_FALLBACK'
    assert report['communication']['sections'] == {}
    assert len(scenario.requests) == 2 and report['tool_attempts'] == 1
    assert 'Temperature causes superior' not in json.dumps(report)
    if mode in ('claim','citation','absent'):
        assert report['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
        assert report['validation']['selection_counts']['omitted'] == 0
    else:
        assert report['synthesis']['status'] == 'INCOMPLETE'


def test_context_compaction_is_lossless_and_reduces_equivalent_serialized_requests(monkeypatch):
    original = agent.compact_model_evidence
    monkeypatch.setattr(agent, 'compact_model_evidence', lambda event,facts,messages:model_evidence(event,facts))
    old = CommunicationScenario(); before = agent.run_investigation(old.question, backend(old))['result']
    monkeypatch.setattr(agent, 'compact_model_evidence', original)
    new = CommunicationScenario(); after = agent.run_investigation(new.question, backend(new))['result']
    assert before['facts'] == after['facts'] and before['validation'] == after['validation']
    assert before['communication'] == after['communication']
    assert expanded_facts(old.requests[-1]['messages']) == expanded_facts(new.requests[-1]['messages'])
    assert new.sizes[0] == old.sizes[0]
    assert new.sizes[-1] < old.sizes[-1] * .9
    old_reply = json.loads(next(m['content'] for m in old.requests[-1]['messages'] if m['role']=='tool'))
    new_reply = json.loads(next(m['content'] for m in new.requests[-1]['messages'] if m['role']=='tool'))
    assert len(old_reply['facts']) == len(new_reply['facts'])
    for old_part,new_part in zip(old_reply['preview']['components'],new_reply['preview']['components']):
        if old_part['name']=='investigate_pet_evidence':
            for key in ('units','partial_reasons','missing_work','source_preview_omissions','comparison_coverage'):
                assert old_part['preview'][key] == new_part['preview'][key]
    assert old_reply['final_selection'] == new_reply['final_selection']


def test_new_context_and_communication_preserve_privacy_and_no_raw_record_display():
    scenario = CommunicationScenario()
    result = investigate_with_backend(scenario.question, backend(scenario))
    view = result['result']
    assert view['communication']['status'] == 'VERIFIED_EXTRACTIVE'
    outbound = json.dumps(scenario.requests)
    for private in ('"rows"','amino_acid_sequence','normalized_sequence','"embedding"','"vector"','/Users/','MODEL_API_KEY'):
        assert private not in outbound
    display = json.dumps(view['communication'])
    assert 'PAZy publication-derived assay:' not in display
    assert all(s not in display for s in ('"record":','amino_acid_sequence','/Users/'))
    assert all(len(json.dumps(p).encode()) < 262144 for p in scenario.requests)


@pytest.mark.parametrize('mode', ['valid','claim'])
def test_apptest_verified_display_or_deterministic_fallback_transient_and_no_rerun(mode, monkeypatch):
    scenario = CommunicationScenario(mode=mode)
    result = investigate_with_backend(scenario.question, backend(scenario))
    calls = []
    def local_result(*args, **kwargs): calls.append(1); return deepcopy(result)
    def blocked(*args, **kwargs): pytest.fail('AppTest cannot call the provider or Keychain')
    monkeypatch.setattr(ui_adapter, 'run_offline', local_result)
    monkeypatch.setattr('zymecycle.ui_live.subprocess.Popen', blocked)
    monkeypatch.setattr('zymecycle.meta.MetaModelClient.complete', blocked)
    at = AppTest.from_file(APP,default_timeout=30).run()
    at.button(key='example_7').click().run()
    at.button(key='investigate').click().run()
    assert not at.exception and calls == [1]
    assert not at.json and not at.dataframe and not at.get('download_button')
    assert 'view' not in at.session_state
    assert any('Evidence → next experiment' in h.value for h in at.subheader)
    if mode == 'valid':
        assert any('Grounded researcher explanation' in h.value for h in at.subheader)
        assert any('not live Muse acceptance' in c.value for c in at.caption)
        assert any('canonical citation' in c.value for c in at.caption)
    else:
        assert not any('Grounded researcher explanation' in h.value for h in at.subheader)
        assert any('not verified' in c.value for c in at.caption)
    assert 'Temperature causes superior' not in ' '.join(t.value for t in at.text)
    at.run()
    assert not at.exception and calls == [1] and not at.text and 'view' not in at.session_state


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get('ZYMECYCLE_REAL_ESM') != '1',reason='Real local ESM-2 opt-in required')
def test_real_esm2_communication_is_computed_not_activity():
    scenario = CommunicationScenario('ESM-2 similarity')
    report = agent.run_investigation(scenario.question, backend(scenario))['result']
    assert report['communication']['status'] == 'VERIFIED_EXTRACTIVE'
    evidence = report['communication']['sections']['evidence']
    assert all(s['evidence_category']=='COMPUTED' for s in evidence)
    assert any('facebook/esm2_t12_35M_UR50D' in s['text'] for s in evidence)
