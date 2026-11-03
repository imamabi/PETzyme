from copy import deepcopy
import json
import pytest
from test_evidence import A, B, observations, documented
from zymecycle.advisor_candidates import generate_experiment_candidates
from zymecycle.conditions import investigate_conditions


@pytest.fixture
def investigation(observations):
    return deepcopy(investigate_conditions(observations, A, B, enzyme_identifier='E1')['result'])


def test_candidates_reuse_gaps_and_exact_measured_cells(investigation):
    original = deepcopy(investigation)
    result = generate_experiment_candidates(investigation)
    assert result['status'] == 'SUCCESS'
    candidates = result['result']
    codes = {c['code'] for c in candidates}
    assert {'VERIFY_DEFINITION_AND_UNITS', 'VERIFY_PROTOCOL_AND_GROUP', 'VERIFY_DETECTION_LIMIT',
            'VERIFY_REPLICATION_AND_UNCERTAINTY', 'CONTROLLED_COMPARISON', 'INDEPENDENT_REPLICATION'} <= codes
    assert all(c['evidence_category'] == 'HYPOTHETICAL' for c in candidates)
    assert {b['statement'] for b in candidates[0]['evidence_basis'] if b['category'] == 'MEASURED'} == {
        'E1: recorded response 0; units UNKNOWN/UNVERIFIED.', 'E1: recorded response 2; units UNKNOWN/UNVERIFIED.'}
    assert '#record=2&column=activity_at_6.5_40_cryPow' in candidates[0]['evidence_basis'][0]['sources'][0]['locator']
    assert investigation == original
    assert result == generate_experiment_candidates(investigation)
    json.dumps(result, allow_nan=False)


def test_missing_forms_remain_separate_and_missing_not_zero(observations):
    powder = {**A, 'temperature_c': 50, 'ph': 8}
    film = {**powder, 'pet_form': 'AMORPHOUS_FILM'}
    inv = investigate_conditions(observations, powder, film, enzyme_identifier='E1')['result']
    result = generate_experiment_candidates(inv)['result']
    assert sum(c['code'] == 'SEPARATE_PET_FORMS' for c in result) == 1
    assert 'Confirm' in next(c['proposed_action'] for c in result if c['code'] == 'SEPARATE_PET_FORMS')
    measurements = [c for c in result if c['code'] == 'MEASURE_EXACT_CONDITION']
    assert [c['target_conditions'] for c in measurements] == [[powder], [film]]
    assert all(not any(b['category'] == 'MEASURED' for b in c['evidence_basis']) for c in result)
    assert not any(c['code'] == 'CONTROLLED_COMPARISON' for c in result)


def test_duplicate_missing_condition_deduplicated(observations):
    absent = {**A, 'temperature_c': 50}
    inv = investigate_conditions(observations, absent, absent, enzyme_identifier='E1')['result']
    result = generate_experiment_candidates(inv)['result']
    assert sum(c['code'] == 'MEASURE_EXACT_CONDITION' for c in result) == 1
    assert len({c['action_id'] for c in result}) == len(result)


def test_no_response_still_requires_real_protein_identity(observations):
    absent = {**A, 'temperature_c': 50}
    known = investigate_conditions(observations, absent, absent, enzyme_identifier='E1')['result']
    assert known['record_identity']['source_row'] == 2
    assert known['record_identity']['csv_sha256'] == observations[0]['csv_sha256']
    generated = generate_experiment_candidates(known)
    assert generated['status'] == 'SUCCESS'
    assert generated['result'][0]['evidence_basis'][0]['csv_sha256'] == observations[0]['csv_sha256']
    unknown = investigate_conditions(observations, absent, absent, enzyme_identifier='UNKNOWN')['result']
    assert unknown['record_identity'] is None
    assert generate_experiment_candidates(unknown)['status'] == 'ERROR'


def test_verified_evidence_does_not_invent_blockers(observations):
    inv = investigate_conditions(observations, A, B, enzyme_identifier='E1', metadata=documented(observations))['result']
    codes = {c['code'] for c in generate_experiment_candidates(inv)['result']}
    assert 'VERIFY_DEFINITION_AND_UNITS' not in codes
    assert 'VERIFY_DETECTION_LIMIT' not in codes
    assert 'REASSESS_COMPARABILITY' not in codes


@pytest.mark.parametrize('change', ['identity', 'conditions', 'source', 'ambiguous', 'record_identity', 'inconsistent_condition'])
def test_invalid_investigations_fail_closed(investigation, change):
    if change == 'identity': investigation['enzyme_identifier'] = None
    if change == 'conditions': investigation['conditions'][0]['condition']['ph'] = float('nan')
    if change == 'source': investigation['source_provenance'] = []
    if change == 'ambiguous': investigation['conditions'][1]['observations'][0]['source_row'] += 1
    if change == 'record_identity': investigation['record_identity']['source_row'] += 1
    if change == 'inconsistent_condition': investigation['conditions'][0]['observations'][0]['ph'] = 7
    assert generate_experiment_candidates(investigation)['status'] == 'ERROR'
