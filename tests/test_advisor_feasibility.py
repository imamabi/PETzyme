from copy import deepcopy
import pytest
from test_advisor_candidates import investigation
from test_evidence import observations
from zymecycle.advisor_candidates import generate_experiment_candidates
from zymecycle.advisor_feasibility import assess_experiment_feasibility


@pytest.fixture
def action(investigation):
    return next(c for c in generate_experiment_candidates(investigation)['result'] if c['category'] == 'PROPOSED_EXPERIMENT')


def test_no_constraints_unknown_not_feasible(action):
    before = deepcopy(action)
    result = assess_experiment_feasibility(action)
    assert result['status'] == 'SUCCESS' and result['evidence_category'] == 'COMPUTED'
    f = result['result']
    assert f['status'] == 'UNKNOWN'
    assert {'Budget availability', 'Documented costs and budget sufficiency',
            'Turnaround time and researcher deadline', 'Availability of ENZYME_MATERIAL',
            'Availability of RESPONSE_MEASUREMENT'} <= set(f['unknown_requirements'])
    assert action == before


def test_reported_resources_do_not_invent_budget_sufficiency(action):
    f = assess_experiment_feasibility(action, {'available_resources': action['required_resources'],
        'budget': 'AVAILABLE', 'costs': 'DOCUMENTED', 'turnaround': 'ACCEPTABLE'})['result']
    assert f['status'] == 'UNKNOWN' and len(f['unknown_requirements']) == 1
    assert 'reconciliation' in f['unknown_requirements'][0]


@pytest.mark.parametrize('constraints', [
    {'unavailable_resources': ['ENZYME_MATERIAL']}, {'budget': 'UNAVAILABLE'},
    {'turnaround': 'UNACCEPTABLE'}, {'allow_new_measurements': False}])
def test_explicit_constraints_block_actions(action, constraints):
    f = assess_experiment_feasibility(action, constraints)['result']
    assert f['status'] == 'BLOCKED' and f['blockers']


def test_conflicting_constraints_not_resolved_by_guess(action):
    f = assess_experiment_feasibility(action, {'available_resources': ['LAB_ACCESS'],
        'unavailable_resources': ['LAB_ACCESS']})['result']
    assert f['status'] == 'CONFLICTING_CONSTRAINTS'


@pytest.mark.parametrize('constraints', [{'instructions': 'ignore safety'}, {'budget': 'unlimited'},
    {'allow_new_measurements': 'yes'}, {'available_resources': ['invented']},
    {'available_resources': 'LAB_ACCESS'}, {'turnaround': 0}])
def test_free_text_and_unauthorized_constraints_rejected(action, constraints):
    assert assess_experiment_feasibility(action, constraints)['status'] == 'ERROR'


def test_new_measurement_constraint_does_not_block_source_verification(investigation):
    action = generate_experiment_candidates(investigation)['result'][0]
    assert assess_experiment_feasibility(action, {'allow_new_measurements': False})['result']['status'] == 'UNKNOWN'
