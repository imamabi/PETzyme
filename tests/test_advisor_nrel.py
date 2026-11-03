"""Real local source checks; the pretrained representation integration is explicitly opt-in."""
import os
import pytest
from zymecycle.advisor import recommend_next_experiments
from zymecycle.conditions import investigate_conditions
from zymecycle.observations import load_nrel, parse_nrel
from scripts.demo_build05 import demonstrate_scenario

A = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}


def test_real_original_values_locators_and_assay_caveats():
    table = load_nrel()['result']
    observations = parse_nrel(table)['result']['observations']
    investigation = investigate_conditions(observations, A, {**A, 'temperature_c': 60}, enzyme_identifier='ESM050')['result']
    assert investigation['comparability']['category'] == 'LIMITED_COMPARISON'
    plan = recommend_next_experiments(investigation)['result']
    measured = [b for b in plan['recommendations'][0]['evidence_basis'] if b['category'] == 'MEASURED']
    assert len(measured) == 2
    assert '0.08499999999999999' in measured[0]['statement'] and '0.13' in measured[1]['statement']
    assert all(b['csv_sha256'] == table['csv_sha256'] for b in measured)
    assert all('record=160&column=activity_at_6.5_' in b['sources'][0]['locator'] for b in measured)
    assert not any(c['code'] == 'VERIFY_DETECTION_LIMIT' for c in plan['recommendations'])
    comparison = next(c for c in plan['recommendations'] if c['code'] == 'CONTROLLED_COMPARISON')
    assert comparison['varied_variables'] == ['temperature']
    assert 'temperature' not in comparison['controlled_variables']
    assert 'pH and buffer' in comparison['controlled_variables']
    assert all(c['feasibility']['status'] == 'UNKNOWN' for c in plan['recommendations'])


def test_real_missing_conditions_are_not_measurements():
    table = load_nrel()['result']
    observations = parse_nrel(table)['result']['observations']
    powder = {**A, 'temperature_c': 50, 'ph': 8}
    film = {**powder, 'pet_form': 'AMORPHOUS_FILM'}
    investigation = investigate_conditions(observations, powder, film, enzyme_identifier='ESM050')['result']
    plan = recommend_next_experiments(investigation)['result']
    assert len(plan['recommendations']) == 4
    assert all(b['category'] == 'COMPUTED' and b['csv_sha256'] == table['csv_sha256']
               for c in plan['recommendations'] for b in c['evidence_basis'])
    measurements = [c for c in plan['recommendations'] if c['code'] == 'MEASURE_EXACT_CONDITION']
    assert {c['target_conditions'][0]['pet_form'] for c in measurements} == {'CRYSTALLINE_POWDER', 'AMORPHOUS_FILM'}


def test_ambiguous_identifier_is_not_a_new_experiment_target():
    observations = parse_nrel(load_nrel()['result'])['result']['observations']
    observation = next(o for o in observations if o['enzyme_identifier'] == 'ESM050')
    duplicate = {**observation, 'source_row': observation['source_row'] + 1000}
    investigated = investigate_conditions([observation, duplicate], A, {**A, 'temperature_c': 60}, enzyme_identifier='ESM050')
    assert investigated['status'] == 'ERROR'
    assert recommend_next_experiments(investigated['result'])['status'] == 'ERROR'


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get('ZYMECYCLE_REAL_ESM') != '1', reason='Explicit real-model smoke opt-in required')
def test_real_pretrained_representation_to_advisor_agent():
    result = demonstrate_scenario('C')
    assert result['status'] == 'SUCCESS', result['validation']['issues']
    assert result['model_requests'] == 4 and result['scientific_tool_attempts'] == 3
    search = result['tool_sequence'][1]['preview']
    assert search['model']['checkpoint'] == 'facebook/esm2_t12_35M_UR50D'
    assert search['coverage']['searched_candidates'] == 8
    assert search['coverage']['outside_window_records'] == 204
    assert len(search['neighbors']) == 5
    assert result['proposals'][-1]['category'] == 'PROPOSED_EXPERIMENT'
    assert not result['observed_facts']
    for action in result['proposals']:
        assert 'target_conditions' in action and action['target_conditions'] == []
        assert 'unknown' in action['expected_information'].lower() or action['category'] == 'EVIDENCE_VERIFICATION'
        assert all(b['category'] == 'COMPUTED' for b in action['evidence_basis'])
        assert all(any('layer=12' in str(loc) for loc in b['locators']) for b in action['evidence_basis'])
