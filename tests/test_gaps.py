import json
from test_evidence import A, B, observations, documented
from petzyme.conditions import investigate_conditions
from petzyme.gaps import identify_evidence_gaps


def test_ordered_needs_no_fabricated_outcomes(observations):
    investigation = investigate_conditions(observations, A, B, enzyme_identifier='E2')['result']
    result = identify_evidence_gaps(investigation)
    gaps = result['result']
    codes = [gap['code'] for gap in gaps]
    assert codes[0] == 'VERIFY_DEFINITION_AND_UNITS'
    assert codes[-1] == 'MEASURE_REQUESTED_CONDITION'
    assert gaps[-1]['target_conditions'] == [B]
    assert [gap['priority'] for gap in gaps] == list(range(1, len(gaps) + 1))
    assert result['provenance'] == investigation['source_provenance']
    assert result['evidence_category'] == 'COMPUTED'
    assert not any(key in gap for gap in gaps for key in ('predicted_value', 'expected_outcome', 'kinetic_parameter'))
    json.dumps(result, allow_nan=False)


def test_zero_detection_requirement(observations):
    investigation = investigate_conditions(observations, A, B, enzyme_identifier='E1')['result']
    assert 'VERIFY_DETECTION_LIMIT' in [g['code'] for g in identify_evidence_gaps(investigation)['result']]


def test_fully_documented_no_invented_gap(observations):
    investigation = investigate_conditions(observations, A, B, enzyme_identifier='E1', metadata=documented(observations))['result']
    assert identify_evidence_gaps(investigation)['result'] == []


def test_form_and_multiple_variable_requirements(observations):
    condition = {**B, 'pet_form': 'AMORPHOUS_FILM'}
    investigation = investigate_conditions(observations, A, condition, enzyme_identifier='E1')['result']
    codes = [g['code'] for g in identify_evidence_gaps(investigation)['result']]
    assert 'SEPARATE_PET_FORMS' in codes and 'ISOLATE_REQUESTED_VARIABLE' in codes


def test_missing_duplicate_request_produces_one_measurement_need(observations):
    absent = {**A, 'temperature_c': 50, 'ph': 8}
    investigation = investigate_conditions(observations, absent, absent, enzyme_identifier='E1')['result']
    gaps = identify_evidence_gaps(investigation)['result']
    assert len([g for g in gaps if g['code'] == 'MEASURE_REQUESTED_CONDITION']) == 1


def test_same_cell_clarifies_independent_comparison(observations):
    investigation = investigate_conditions(observations, A, A, enzyme_identifier='E1', metadata=documented(observations))['result']
    codes = [g['code'] for g in identify_evidence_gaps(investigation)['result']]
    assert codes == ['CLARIFY_COMPARISON_TARGET']
