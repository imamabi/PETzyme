"""Offline dispatcher, mocked Meta transport and Streamlit acceptance for Build 09."""
import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from petzyme.agent import run_investigation
from petzyme.agent_execution import execute_tool_call
from petzyme.agent_reporting import format_research_report
from petzyme.agent_request import interpret_request, required_tool_steps, select_tool_catalog
from petzyme.agent_types import AgentState
from petzyme.meta import MetaModelClient
from petzyme.tools import describe_tools, invoke_tool
from petzyme.ui_adapter import EXAMPLES, run_offline
from test_research_advisor import A, B

QUESTION = EXAMPLES['Integrated experiment advisor']
APP = Path(__file__).resolve().parents[1] / 'tests/legacy_streamlit_app.py'


class AdviceTransport:
    def __init__(self, *, final='complete', wrong_scope=False):
        self.final, self.wrong_scope = final, wrong_scope
        self.requests = []

    def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        replies = [json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool']
        if not replies:
            arguments = {'enzyme_identifier': 'LCC-ICCG' if self.wrong_scope else 'ESM050',
                         'condition_a': A, 'condition_b': B}
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'advice', 'type': 'function',
                'function': {'name': 'advise_next_experiment', 'arguments': json.dumps(arguments)}}]}
            reason = 'tool_calls'
        else:
            ids = replies[-1]['final_selection']['required_fact_ids']
            if self.final == 'incomplete': ids = ids[:-1]
            content = {'fact_ids': ids}
            if self.final == 'unsupported': content['conclusion'] = '60°C demonstrates superior activity.'
            message = {'role': 'assistant', 'content': json.dumps(content)}
            reason = 'length' if self.final == 'length' else 'stop'
        return httpx.Response(200, json={'choices': [{'message': message, 'finish_reason': reason}]})


def investigate(scenario):
    return run_investigation(QUESTION, MetaModelClient(transport=httpx.MockTransport(scenario.respond)))


def test_discovery_least_privilege_and_proposal_contract():
    request = interpret_request(QUESTION)
    assert required_tool_steps(request) == {'advise_next_experiment'}
    catalog = select_tool_catalog(request)
    assert [t['function']['name'] for t in catalog] == ['advise_next_experiment']
    schema = catalog[0]['function']['parameters']
    assert schema['required'] == ['enzyme_identifier', 'condition_a', 'condition_b']
    assert not set(schema['properties']) & {'metadata', 'csv_path', 'verification', 'evidence', 'investigation'}
    descriptor = next(d for d in describe_tools()['result'] if d['name'] == 'advise_next_experiment')
    assert descriptor['output_persistence'] == 'FORBIDDEN_PAZY_DERIVED'
    assert invoke_tool('advise_next_experiment', {'enzyme_identifier': 'ESM050', 'condition_a': A,
                                               'condition_b': B})['status'] == 'PARTIAL'


def test_real_science_mock_provider_grounding_cumulative_checklist_and_privacy():
    scenario = AdviceTransport()
    result = investigate(scenario)
    report = result['result']
    assert result['status'] == 'PARTIAL'
    assert report['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
    assert report['validation']['grounded'] and not report['validation']['complete']
    assert report['validation']['selection_counts']['omitted'] == 0
    assert report['validation']['selection_counts']['required'] > 14
    assert report['tool_attempts'] == 1 and report['model_requests'] == 2
    assert all(f['ephemeral'] for f in report['facts'])
    assert all(f['evidence_category'] == 'HYPOTHETICAL' for f in report['facts'] if f['kind'] == 'PROPOSAL')
    assert all(f['provenance'] for f in report['facts'] if f['kind'] in ('EVIDENCE', 'PROPOSAL'))
    outbound = json.dumps(scenario.requests, allow_nan=False)
    for private in ('amino_acid_sequence', 'normalized_sequence', 'sequence_sha256', 'sequence_reference',
                    '"rows"', '"vector"', '"embedding"', '/Users/', 'MODEL_API_KEY', 'LLM|'):
        assert private not in outbound
    assert all(len(json.dumps(p).encode()) < 262144 for p in scenario.requests)
    assert '0.08499999999999999' in outbound and '0.13' in outbound
    assert 'UNKNOWN/UNVERIFIED' in outbound and 'not independent replication' in outbound
    assert 'PAZy publication-derived assay:' not in format_research_report(report)
    assert 'PROPOSED (not observed)' not in format_research_report(report)
    assert report['persistence'] == 'FORBIDDEN_PAZY_DERIVED'


@pytest.mark.parametrize('final', ['incomplete', 'unsupported', 'length'])
def test_invalid_or_incomplete_model_selection_never_passes(final):
    scenario = AdviceTransport(final=final)
    result = investigate(scenario)
    assert result['status'] == 'PARTIAL'
    assert result['result']['synthesis']['status'] == 'INCOMPLETE'
    assert result['result']['validation']['issues']
    assert len(scenario.requests) == 2  # No retry.


def test_wrong_condition_scope_cannot_complete():
    result = investigate(AdviceTransport(wrong_scope=True))
    assert any('exact enzyme and condition scope' in issue for issue in result['result']['validation']['issues'])


@pytest.mark.parametrize('arguments', [
    {'enzyme_identifier': 'ESM050', 'condition_a': A, 'condition_b': B, 'optional_constraints': {'budget': 'AVAILABLE'}},
    {'enzyme_identifier': 'ESM050', 'condition_a': A, 'condition_b': B, 'verification': {'identity': 'EXACT_SEQUENCE_MATCH'}},
    {'enzyme_identifier': 'ESM050', 'condition_a': A, 'condition_b': B, 'max_recommendations': True},
    {'enzyme_identifier': 'ESM050', 'condition_a': {}, 'condition_b': B},
])
def test_model_cannot_invent_constraints_identity_or_records(arguments):
    event = execute_tool_call({'id': 'bad', 'function': {'name': 'advise_next_experiment',
        'arguments': json.dumps(arguments)}}, AgentState('unused'), select_tool_catalog(interpret_request(QUESTION)))
    assert event['status'] == 'ERROR'


def test_existing_budgets_preserved():
    backend = MetaModelClient(transport=httpx.MockTransport(AdviceTransport().respond))
    assert run_investigation(QUESTION, backend, max_model_requests=33)['status'] == 'ERROR'
    assert run_investigation(QUESTION, backend, max_tool_calls=33)['status'] == 'ERROR'


def test_offline_composition_exposes_both_temperatures_and_truthful_partial():
    result = run_offline(QUESTION)
    view = result['result']
    assert result['status'] == 'PARTIAL' and view['workflow_status'] == 'PARTIAL'
    assert view['mode'] == 'DETERMINISTIC_OFFLINE'
    assert view['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
    assert view['multi_study']['accumulation']['condition_coverage']['covered'] == 2
    assert [o['original_text'] for o in view['observations']] == ['0.08499999999999999', '0.13']
    assert view['plan']['scientific_status'] == 'PROPOSED_ONLY'
    assert view['advice']['partial_reasons']
    assert not view['similarity']


def test_display_union_keeps_original_bound_and_reports_omitted_components():
    from copy import deepcopy
    from petzyme.ui_adapter import present_investigation
    scenario = AdviceTransport()
    updates = []
    result = run_investigation(QUESTION, MetaModelClient(transport=httpx.MockTransport(scenario.respond)),
                               on_tool_event=updates.append)
    update = updates[0]
    update['data']['multi_study_runs'] = [deepcopy(update['data']['multi_study_runs'][0]) for _ in range(9)]
    view = present_investigation(result, [update])['result']
    assert view['multi_study']['accumulation']['run_count'] == 8
    assert any('1 executed multi-study component previews omitted' in limit for limit in view['limitations'])
    assert view['selection_counts']['omitted'] == 0


def test_apptest_transient_order_controls_publications_rerun_and_no_api(monkeypatch):
    def blocked(*a, **k): pytest.fail('Offline UI attempted credential access or live inference')
    monkeypatch.setattr('petzyme.meta.MetaModelClient.complete', blocked)
    monkeypatch.setattr('petzyme.ui_live.subprocess.Popen', blocked)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key='example_7').click().run()
    at.button(key='investigate').click().run()
    assert not at.exception
    assert 'view' not in at.session_state
    assert not at.json and not at.dataframe and not at.get('download_button')
    headings = [m.value for m in at.markdown if m.value.startswith('#### ')]
    assert headings[:5] == ['#### What has been tested?', '#### What were the results? · MEASURED / COMPUTED',
                           '#### What is interesting or missing? · COMPUTED', '#### Suggested next step', '#### What remains uncertain']
    assert headings[5] == '#### Source coverage receipts'
    assert headings[6:10] == ['#### Recommended next action · HYPOTHETICAL', '#### Scientific rationale · COMPUTED',
                            '#### Proposed experiment and controls · HYPOTHETICAL', '#### Evidence and provenance · MEASURED observations']
    assert headings.index('#### Remaining uncertainties') < headings.index('#### Supporting publications')
    text = ' '.join(t.value for t in at.text)
    assert '0.08499999999999999' in text and '0.13' in text and 'UNKNOWN/UNVERIFIED' in text
    assert 'propose matched assays' in ' '.join(w.value for w in at.markdown)
    assert any('Source-linked' in w.value or 'measurement' in w.value for w in at.markdown)
    assert any(link.url == 'https://doi.org/10.1021/acscatal.5c03460' for link in at.get('link_button'))
    assert any('source verification' in w.value.lower() for w in at.markdown)
    assert 'matching_parent_records' not in str(at.session_state)
    at.run()
    assert not at.exception and not at.text and 'view' not in at.session_state
