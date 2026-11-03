from copy import deepcopy
import json
import pytest

from petzyme.evidence import assess_evidence
from petzyme.experimental import PROTOCOL_FIELDS, cell_locator
from petzyme.observations import load_nrel, parse_nrel

A = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
B = {**A, 'temperature_c': 60}


@pytest.fixture
def observations(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow,activity_at_6.5_60_cryPow,activity_at_6.5_40_aFilm\n'
                    'E1,ACDEF,0,2,3\nE2,ACDEG,4,,5\n')
    return parse_nrel(load_nrel(str(path))['result'])['result']['observations']


def documented(observations):
    return [{'cell_locator': cell_locator(o), 'csv_sha256': o['csv_sha256'],
             'documentation': [{**o['source'], 'locator': 'synthetic-protocol#explicit-cell-map'}],
             'details': {'experimental_group': 'synthetic-group', 'measurement_definition': 'synthetic-response',
                         'measurement_unit': 'synthetic-unit', 'units_verified': True,
                         'protocol_id': 'synthetic-protocol', 'protocol': {key: 'synthetic-fixed' for key in PROTOCOL_FIELDS},
                         'detection_limit': 'synthetic documented threshold', 'replicates': 2,
                         'uncertainty': 'synthetic documented uncertainty', 'publication_doi': 'synthetic-publication',
                         'publication_mapping_verified': True}} for o in observations]


def test_raw_limitations_and_no_procedure_absence(observations):
    before = deepcopy(observations)
    result = assess_evidence(observations, expected_conditions=[A, B])
    codes = {i['code'] for i in result['result']['issues']}
    assert codes == {'UNITS_UNVERIFIED', 'MISSING_EXPERIMENTAL_METADATA', 'ZERO_WITHOUT_DETECTION_LIMIT',
                     'REPLICATES_UNREPORTED', 'UNCERTAINTY_UNREPORTED', 'PUBLICATION_MAPPING_UNVERIFIED',
                     'INCOMPLETE_CONDITION_COVERAGE'}
    gap = result['result']['missing_conditions'][0]
    assert len(result['result']['missing_conditions']) == 1
    assert gap['enzyme_identifier'] == 'E2' and gap['condition'] == B
    assert gap['source_row'] == 3 and gap['csv_sha256'] == observations[3]['csv_sha256']
    assert observations == before
    assert all(o['evidence_category'] == 'MEASURED' for o in observations)
    assert result['evidence_category'] == 'COMPUTED'
    assert all(issue['provenance'] for issue in result['result']['issues'])
    json.dumps(result, allow_nan=False)


def test_documented_cells_and_explicit_coverage(observations):
    selected = observations[:2]
    result = assess_evidence(selected, documented(selected), [A, B])
    assert result['result']['issues'] == []


@pytest.mark.parametrize('change', ['no_documentation', 'different_hash', 'different_locator'])
def test_unbound_assertions_do_not_verify_units(observations, change):
    record = documented(observations[:1])[0]
    if change == 'no_documentation': record['documentation'] = []
    if change == 'different_hash': record['csv_sha256'] = 'other'
    if change == 'different_locator': record['cell_locator'] = 'other'
    result = assess_evidence(observations[:1], [record])
    assert 'UNITS_UNVERIFIED' in [i['code'] for i in result['result']['issues']]


def test_zero_specific_and_duplicates_not_replicates(observations):
    result = assess_evidence([observations[0], observations[0]])
    assert result['result']['observation_count'] == 1
    issue = next(i for i in result['result']['issues'] if i['code'] == 'ZERO_WITHOUT_DETECTION_LIMIT')
    assert issue['affected_cells'] == [cell_locator(observations[0])]
    assert 'COVERAGE_SCOPE_UNSPECIFIED' in [i['code'] for i in result['result']['issues']]


def test_invalid_context_and_conflicts(observations):
    record = documented(observations[:1])[0]
    assert assess_evidence(observations, [record, record])['status'] == 'ERROR'
    record['details']['replicates'] = 0
    assert assess_evidence(observations, [record])['status'] == 'ERROR'
    assert assess_evidence([observations[0], {**observations[0], 'original_text': '9'}])['status'] == 'ERROR'
    assert assess_evidence(observations, expected_conditions=[{**A, 'ph': float('nan')}])['status'] == 'ERROR'


def test_empty_is_not_zero():
    result = assess_evidence([], expected_conditions=[A])
    assert result['status'] == 'NO_DATA'
    assert result['result']['observation_count'] == 0
    assert 'NO_OBSERVATIONS' in [i['code'] for i in result['result']['issues']]


@pytest.mark.parametrize('field,value', [('measurement_unit', ' '), ('units_verified', 'true'),
                                        ('protocol', {'unknown': 'value'}), ('replicates', True)])
def test_malformed_documented_claims_are_rejected(observations, field, value):
    records = documented(observations[:1])
    records[0]['details'][field] = value
    assert assess_evidence(observations[:1], records)['status'] == 'ERROR'


def test_coverage_preserves_distinct_source_identities(observations):
    original = observations[3]
    alternate = {**original, 'source': {**original['source'], 'doi': 'synthetic-other-source'}}
    result = assess_evidence([original, alternate], expected_conditions=[B])
    gaps = result['result']['missing_conditions']
    assert len(gaps) == 2
    assert gaps[0]['source']['doi'] != gaps[1]['source']['doi']


def test_dataset_doi_is_not_a_verified_publication_mapping(observations):
    records = documented(observations[:1])
    records[0]['details']['publication_doi'] = observations[0]['source']['doi']
    result = assess_evidence(observations[:1], records)
    assert 'PUBLICATION_MAPPING_UNVERIFIED' in [issue['code'] for issue in result['result']['issues']]
