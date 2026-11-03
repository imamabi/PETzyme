import json
from test_evidence import A, B, observations, documented
from petzyme.conditions import investigate_conditions


def test_single_variable_observed_but_no_causal_claim(observations):
    result = investigate_conditions(observations, A, B, enzyme_identifier='E1')
    value = result['result']
    assert result['status'] == 'SUCCESS'
    assert value['changed_variables'] == ['temperature_c']
    assert value['comparability']['category'] == 'LIMITED_COMPARISON'
    assert not value['direct_comparison_supported'] and not value['causal_attribution_supported']
    assert [entry['observations'][0]['original_text'] for entry in value['conditions']] == ['0', '2']
    assert value['conditions'][0]['observations'][0]['evidence_category'] == 'MEASURED'
    assert result['evidence_category'] == 'COMPUTED'
    assert not any(key in value for key in ('prediction', 'delta', 'ranking', 'causal_effect'))
    json.dumps(result, allow_nan=False)


def test_documented_controlled_comparison(observations):
    result = investigate_conditions(observations, A, B, enzyme_identifier='E1', metadata=documented(observations))
    assert result['result']['direct_comparison_supported'] is True
    assert result['result']['causal_attribution_supported'] is False


def test_missing_measurement_is_not_zero(observations):
    result = investigate_conditions(observations, A, B, enzyme_identifier='E2')
    assert result['status'] == 'PARTIAL'
    assert result['result']['conditions'][1] == {'condition': B, 'observations': [], 'status': 'MISSING'}
    assert not result['result']['direct_comparison_supported']
    gap = result['result']['evidence']['missing_conditions'][0]
    assert gap['enzyme_identifier'] == 'E2' and gap['condition'] == B and gap['source_row'] == 3


def test_no_data_and_selected_record_provenance(observations):
    absent = {**A, 'temperature_c': 50, 'ph': 8}
    result = investigate_conditions(observations, absent, absent, selected_record=observations[0])
    assert result['status'] == 'NO_DATA'
    assert result['provenance']
    assert all(entry['observations'] == [] for entry in result['result']['conditions'])
    assert investigate_conditions(observations, A, B, enzyme_identifier='UNKNOWN')['status'] == 'NO_DATA'


def test_pet_forms_and_multiple_changes(observations):
    film = {**A, 'pet_form': 'AMORPHOUS_FILM'}
    result = investigate_conditions(observations, A, film, enzyme_identifier='E1', metadata=documented(observations))
    assert result['result']['comparability']['category'] == 'NOT_COMPARABLE'
    assert 'pet_form' in result['result']['changed_variables']
    result = investigate_conditions(observations, A, {**B, 'ph': 8}, enzyme_identifier='E1')
    assert result['result']['changed_variables'] == ['temperature_c', 'ph']
    assert not result['result']['direct_comparison_supported']


def test_same_condition_not_counted_as_independent_measurements(observations):
    result = investigate_conditions(observations, A, A, enzyme_identifier='E1')
    assert result['result']['comparability']['observation_count'] == 1
    assert not result['result']['direct_comparison_supported']


def test_invalid_or_ambiguous_selection(observations):
    assert investigate_conditions(observations, A, B)['status'] == 'ERROR'
    assert investigate_conditions(observations, A, B, enzyme_identifier='E1', selected_record=observations[0])['status'] == 'ERROR'
    assert investigate_conditions(observations, {**A, 'ph': float('nan')}, B, enzyme_identifier='E1')['status'] == 'ERROR'
    duplicate = {**observations[3], 'enzyme_identifier': 'E1'}
    assert investigate_conditions([*observations, duplicate], A, B, enzyme_identifier='E1')['status'] == 'ERROR'


def test_selected_record_cannot_relabel_source_row(observations):
    selected = {**observations[0], 'enzyme_identifier': 'forged'}
    assert investigate_conditions(observations, A, B, selected_record=selected)['status'] == 'ERROR'
