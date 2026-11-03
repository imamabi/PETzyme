from copy import deepcopy
import json
import pytest
from test_evidence import observations, documented

from petzyme.comparability import check_comparability


def test_same_column_descriptive_but_units_limited(observations):
    selected = [observations[0], observations[3]]
    result = check_comparability(selected)
    assert result['result']['category'] == 'LIMITED_COMPARISON'
    assert result['result']['descriptive_within_column_supported'] is True
    assert result['result']['quantitative_comparison_supported'] is False
    assert 'verified_units' in result['result']['missing_fields']


def test_documented_group_comparable(observations):
    selected = [observations[0], observations[3]]
    before = deepcopy(selected)
    result = check_comparability(selected, documented(selected))
    assert result['result']['category'] == 'COMPARABLE_WITHIN_DOCUMENTED_GROUP'
    assert result['result']['quantitative_comparison_supported'] is True
    assert selected == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field,value', [('temperature_c', 50), ('ph', 8), ('pet_form', 'AMORPHOUS_FILM')])
def test_conditions_differ(observations, field, value):
    selected = [observations[0], {**observations[3], field: value}]
    result = check_comparability(selected, documented(selected))
    assert result['result']['category'] == 'NOT_COMPARABLE'
    assert 'numerical_comparison' not in result['result']
    assert field in result['result']['blocking_differences']
    assert not result['result']['descriptive_within_column_supported']


@pytest.mark.parametrize('field', ['measurement_unit', 'measurement_definition', 'experimental_group', 'protocol_id'])
def test_no_assumed_definition_or_unit_equivalence(observations, field):
    selected = [observations[0], observations[3]]
    records = documented(selected)
    records[1]['details'][field] = 'different'
    assert check_comparability(selected, records)['result']['category'] == 'NOT_COMPARABLE'


def test_available_protocol_differences_block_comparison(observations):
    selected = [observations[0], observations[3]]
    records = documented(selected)
    records[1]['details']['protocol']['buffer'] = 'different buffer'
    result = check_comparability(selected, records)['result']
    assert result['category'] == 'NOT_COMPARABLE'
    assert 'protocol.buffer' in result['blocking_differences']


def test_sources_are_not_equated_by_name(observations):
    selected = [observations[0], deepcopy(observations[3])]
    selected[1]['source']['doi'] = 'other-doi'
    result = check_comparability(selected, documented(selected))['result']
    assert result['category'] == 'NOT_COMPARABLE' and 'source' in result['blocking_differences']


def test_explicit_controlled_variable_not_causal_proof(observations):
    selected = observations[:2]
    result = check_comparability(selected, documented(selected), varying_variable='temperature_c')['result']
    assert result['category'] == 'COMPARABLE_WITHIN_DOCUMENTED_GROUP'
    assert result['differing_fields'] == ['temperature_c']
    assert result['blocking_differences'] == []
    assert not result['descriptive_within_column_supported']
    assert check_comparability(selected, varying_variable='temperature_c')['result']['category'] == 'LIMITED_COMPARISON'
    assert check_comparability(selected, varying_variable='pet_form')['status'] == 'ERROR'


@pytest.fixture
def source_observations(request):
    return request.getfixturevalue('observations')


def test_descriptive_numbers_require_complete_source_bound_comparability(source_observations):
    selected = [{**source_observations[0], 'original_text': '1', 'numerical_value': 1.0, 'is_zero': False}, source_observations[1]]
    result = check_comparability(selected, documented(selected), varying_variable='temperature_c')['result']
    numeric = result['numerical_comparison']
    assert numeric['difference_b_minus_a'] == '1'
    assert numeric['ratio_b_over_a'] == '2'
    assert numeric['percent_change_from_a'] == '100'
    assert numeric['evidence_category'] == 'COMPUTED'
    assert numeric['metadata_basis'] == 'CALLER_SUPPLIED_SOURCE_BOUND_NOT_INDEPENDENTLY_AUTHENTICATED'
    assert 'numerical_comparison' not in check_comparability(selected, varying_variable='temperature_c')['result']
    from petzyme.agent_evidence import project_evidence
    projection, facts = project_evidence('synthetic_compare', 'check_comparability',
                                         check_comparability(selected, documented(selected), varying_variable='temperature_c'))
    assert projection['numerical_comparison']['difference_b_minus_a'] == '1'
    assert any(fact['evidence_category'] == 'COMPUTED' and 'No statistical significance' in fact['text'] for fact in facts)


def test_zero_baseline_never_invents_a_ratio_or_percentage(source_observations):
    selected = source_observations[:2]
    result = check_comparability(selected, documented(selected), varying_variable='temperature_c')['result']
    assert result['numerical_comparison']['ratio_b_over_a'] is None
    assert result['numerical_comparison']['percent_change_from_a'] is None


def test_optional_numerical_result_does_not_make_legacy_investigations_invalid(source_observations):
    from petzyme.conditions import investigate_conditions
    from petzyme.tools import invoke_tool

    a = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
    b = {**a, 'temperature_c': 60}
    investigation = investigate_conditions(source_observations, a, b, enzyme_identifier='E1')['result']
    assert 'numerical_comparison' not in investigation['comparability']
    assert invoke_tool('identify_evidence_gaps', {'investigation': investigation})['status'] == 'SUCCESS'


def test_missing_evidence_and_repeated_record(observations):
    for selected in ([], [observations[0]], [observations[0], observations[0]]):
        result = check_comparability(selected)
        assert result['result']['category'] == 'INSUFFICIENT_INFORMATION'
        assert result['status'] == 'NO_DATA'


def test_varied_temperature_with_different_proteins_is_confounded(observations):
    selected = [observations[1], observations[3]]
    result = check_comparability(selected, documented(selected), varying_variable='temperature_c')['result']
    assert result['category'] == 'NOT_COMPARABLE'
    assert 'protein_record' in result['blocking_differences']
