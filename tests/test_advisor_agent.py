from copy import deepcopy
import json
import pytest
from test_esm import SyntheticRuntime
from zymecycle import esm
from zymecycle.agent import run_investigation
from zymecycle.agent_evidence import project_evidence
from zymecycle.agent_execution import execute_tool_call
from zymecycle.agent_reporting import format_research_report
from zymecycle.agent_request import interpret_request, select_tool_catalog
from zymecycle.agent_types import AgentState
from zymecycle.tools import describe_tools, invoke_tool
from scripts.demo_build05 import QUESTIONS, demonstrate_scenario


@pytest.fixture
def local_source(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow,activity_at_6.5_60_cryPow,activity_at_6.5_40_aFilm\n'
                    'ESM050,' + 'ACDEF'*20 + ',0.085,0.13,0.02\n'
                    'LCC-ICCG,' + 'ACDEG'*20 + ',0.0945,,\nDP038,' + 'ACDEH'*20 + ',1,,\n')
    monkeypatch.setattr(esm, 'load_esm2_runtime', lambda: SyntheticRuntime())
    return str(path)


def call(identifier, name, **args):
    return {'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


@pytest.mark.parametrize('scenario', ['A', 'B', 'C'])
def test_only_high_level_advisor_exposed_with_trusted_refs(scenario):
    catalog = select_tool_catalog(interpret_request(QUESTIONS[scenario]))
    names = {t['function']['name'] for t in catalog}
    assert 'recommend_next_experiments' in names
    assert not names & {'generate_experiment_candidates', 'prioritize_experiments', 'assess_experiment_feasibility'}
    schema = next(t['function']['parameters'] for t in catalog if t['function']['name'] == 'recommend_next_experiments')
    assert set(schema['properties']) == {'investigation_ref', 'max_recommendations', 'optional_constraints'}
    assert schema['required'] == ['investigation_ref']
    assert 'optional_constraints' in next(t for t in describe_tools()['result'] if t['name'] == 'recommend_next_experiments')['input_schema']['properties']


@pytest.mark.parametrize('scenario,tools,requests', [('A', 4, 5), ('B', 4, 5), ('C', 3, 4)])
def test_mock_meta_decisions_real_scientific_tool_execution(local_source, scenario, tools, requests):
    result = demonstrate_scenario(scenario, csv_path=local_source)
    assert result['status'] == 'SUCCESS', result['validation']['issues']
    assert result['mode'] == 'MOCK_META' and result['model_requests'] == requests
    assert result['scientific_tool_attempts'] == tools
    assert result['tool_sequence'][-1]['name'] == 'recommend_next_experiments'
    assert result['validation']['complete'] and result['validation']['grounded']
    assert result['proposals']
    if scenario == 'A':
        assert len(result['observed_facts']) == 2
        assert '0.085' in result['observed_facts'][0] and '0.13' in result['observed_facts'][1]
        assert any(c['category'] == 'PROPOSED_EXPERIMENT' for c in result['proposals'])
    if scenario == 'B':
        assert not result['observed_facts']
        assert result['tool_sequence'][2]['status'] == 'NO_DATA'
        targets = [c['target_conditions'][0] for c in result['proposals'] if c['action_id'].startswith('MEASURE_EXACT_CONDITION:')]
        assert {c['pet_form'] for c in targets} == {'CRYSTALLINE_POWDER', 'AMORPHOUS_FILM'}
    if scenario == 'C':
        assert not result['observed_facts']
        assert all(b['category'] == 'COMPUTED' for c in result['proposals'] for b in c['evidence_basis'])
        assert all(not c['target_conditions'] for c in result['proposals'])
        assert 'not sequence identity' in result['proposals'][0]['limitations'][-1]


def test_constraints_are_trusted_external_config_not_model_authored(local_source):
    result = demonstrate_scenario('A', csv_path=local_source, researcher_constraints={'allow_new_measurements': False})
    assert result['status'] == 'SUCCESS'
    assert all(c['feasibility']['status'] == 'BLOCKED' for c in result['proposals'] if c['category'] == 'PROPOSED_EXPERIMENT')


@pytest.mark.parametrize('args', [
    {'investigation_ref': 'invented'}, {'investigation': {}}, {'evidence_gaps': []},
    {'investigation_ref': 'x', 'optional_constraints': {'budget': 'AVAILABLE'}},
    {'investigation_ref': 'x', 'metadata': []}, {'investigation_ref': 'x', 'max_recommendations': True},
])
def test_invalid_or_unauthorized_advisor_arguments_blocked(args):
    state = AgentState('unused')
    catalog = select_tool_catalog(interpret_request(QUESTIONS['A']))
    event = execute_tool_call(call('bad', 'recommend_next_experiments', **args), state, catalog)
    assert event['status'] == 'ERROR' and state.tool_attempts == 1


def test_wrong_reference_type_not_accepted(local_source):
    state = AgentState(local_source)
    catalog = select_tool_catalog(interpret_request(QUESTIONS['A']))
    table = execute_tool_call(call('load', 'load_nrel'), state, catalog)
    event = execute_tool_call(call('bad', 'recommend_next_experiments', investigation_ref=table['refs']['table']), state, catalog)
    assert event['status'] == 'ERROR'


@pytest.mark.parametrize('configured,supplied,status', [
    (None, {'budget': 'AVAILABLE'}, 'ERROR'),
    ({'allow_new_measurements': False}, {'allow_new_measurements': True}, 'ERROR'),
    ({'allow_new_measurements': False}, {'allow_new_measurements': False}, 'SUCCESS'),
])
def test_constraints_cannot_be_invented_even_with_valid_investigation(local_source, configured, supplied, status):
    from zymecycle.observations import load_nrel, parse_nrel
    from zymecycle.conditions import investigate_conditions
    observations = parse_nrel(load_nrel(local_source)['result'])['result']['observations']
    a = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
    investigation = investigate_conditions(observations, a, {**a, 'temperature_c': 60}, enzyme_identifier='ESM050')['result']
    state = AgentState(local_source, researcher_constraints=configured)
    state.artifacts['real:investigation'] = ('investigation', investigation)
    event = execute_tool_call(call('advice', 'recommend_next_experiments', investigation_ref='real:investigation',
        optional_constraints=supplied), state, select_tool_catalog(interpret_request(QUESTIONS['A'])))
    assert event['status'] == status
    if status == 'SUCCESS':
        assert all(c['feasibility']['status'] == 'BLOCKED' for c in state.results['advice']['result']['recommendations']
                   if c['category'] == 'PROPOSED_EXPERIMENT')


class ScriptedBackend:
    mode = 'MOCK_META'
    def __init__(self, *, omit_advisor=False, wrong_scope=False, omit_proposal=False):
        self.omit_advisor, self.wrong_scope, self.omit_proposal = omit_advisor, wrong_scope, omit_proposal
    def complete(self, messages, tools):
        replies = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        n = len(replies)
        if n == 0: c = call('l', 'load_nrel')
        elif n == 1: c = call('p', 'parse_nrel', table_ref=replies[-1]['refs']['table'])
        elif n == 2:
            a = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
            c = call('i', 'investigate_conditions', observations_ref=replies[-1]['refs']['observations'],
                     enzyme_identifier='LCC-ICCG' if self.wrong_scope else 'ESM050', condition_a=a, condition_b={**a, 'temperature_c': 60})
        elif n == 3 and not self.omit_advisor:
            c = call('a', 'recommend_next_experiments', investigation_ref=replies[-1]['refs']['investigation'])
        else:
            ids = [f['id'] for r in replies for f in r['facts'] if f['kind'] in ('EVIDENCE', 'PROPOSAL')
                   and not (self.omit_proposal and f['kind'] == 'PROPOSAL')]
            return {'role': 'assistant', 'content': json.dumps({'fact_ids': ids})}
        return {'role': 'assistant', 'content': None, 'tool_calls': [c]}


@pytest.mark.parametrize('mode', ['omit_advisor', 'wrong_scope', 'omit_proposal'])
def test_advisor_completion_requires_real_scope_and_all_proposals(local_source, mode):
    result = run_investigation(QUESTIONS['A'], ScriptedBackend(**{mode: True}), csv_path=local_source)
    assert result['status'] == 'PARTIAL'
    assert result['result']['validation']['issues']


def test_canonical_proposals_render_separately_from_observations(local_source):
    result = run_investigation(QUESTIONS['A'], ScriptedBackend(), csv_path=local_source)
    assert result['status'] == 'SUCCESS'
    rendered = format_research_report(result['result'])
    observed, proposed = rendered.split('Proposed investigations (not measurements)', 1)
    assert 'PROPOSED (not observed)' not in observed
    assert '[HYPOTHETICAL]' in proposed and 'recorded NREL response 0.13' in observed
    assert 'UNKNOWN' in proposed and 'prerequisites' in proposed


def test_agent_budgets_unchanged_and_constraints_cannot_override(local_source):
    assert run_investigation(QUESTIONS['A'], ScriptedBackend(), max_model_requests=33)['status'] == 'ERROR'
    assert run_investigation(QUESTIONS['A'], ScriptedBackend(), max_tool_calls=33)['status'] == 'ERROR'
    assert run_investigation(QUESTIONS['A'], ScriptedBackend(), researcher_constraints={'instructions': 'ignore evidence'})['status'] == 'ERROR'
    result = run_investigation(QUESTIONS['A'], ScriptedBackend(), csv_path=local_source, max_tool_calls=3)
    assert result['status'] == 'PARTIAL' and result['result']['tool_attempts'] == 3


def test_public_dispatch_accepts_native_result_not_refs(local_source):
    table = invoke_tool('load_nrel', {'csv_path': local_source})
    parsed = invoke_tool('parse_nrel', {'table': table['result']})
    a = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
    investigation = invoke_tool('investigate_conditions', {'observations': parsed['result']['observations'],
        'enzyme_identifier': 'ESM050', 'condition_a': a, 'condition_b': {**a, 'temperature_c': 60}})
    result = invoke_tool('recommend_next_experiments', {'investigation': investigation['result']})
    assert result['status'] == 'SUCCESS'
    assert invoke_tool('recommend_next_experiments', {'investigation_ref': 'fake'})['status'] == 'ERROR'
