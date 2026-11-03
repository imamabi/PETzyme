import hashlib
import json
from pathlib import Path

import pytest

from zymecycle.observations import (filter_observations, load_nrel, parse_nrel,
                                    select_nrel_protein, summarize_condition)


@pytest.fixture
def table(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence,round,activity_at_6.5_40_cryPow,activity_at_6.5_40_aFilm,activity_at_7.5_60_cryPow\n'
                    'E1,acdef,1,0.00,90,\n'
                    'E2,ACDEG,1, 2.500 ,,3\n'
                    'E3,ACDEH,1,,100,0\n'
                    'E4,ACDEI,1,10,99,4\n'
                    'E5,ACDEK,1,-1,98,5\n'
                    'E6,ACDEL,1,9,,6\n'
                    'E7,ACDEM,1,8,,7\n'
                    'E8,ACDEN,1,7,,8\n')
    return load_nrel(str(path))['result']


def test_original_rows_and_fingerprint(table):
    assert table['rows'][1][3] == ' 2.500 '
    parsed = parse_nrel(table)
    observations = parsed['result']['observations']
    assert parsed['result']['missing_cells'] == 6
    first = observations[0]
    assert first['source_row'] == 2 and first['source_column'] == 'activity_at_6.5_40_cryPow'
    assert first['enzyme_identifier'] == 'E1'
    assert first['original_text'] == '0.00' and first['numerical_value'] == 0
    assert first['is_zero'] is True and first['unit_status'] == 'UNKNOWN/UNVERIFIED'
    assert first['sequence_sha256'] == hashlib.sha256(b'ACDEF').hexdigest()
    assert first['temperature_c'] == 40 and first['ph'] == 6.5
    assert first['pet_form'] == 'CRYSTALLINE_POWDER'
    assert first['source']['doi'] == '10.5281/zenodo.15417757'
    assert first['csv_sha256'] == table['csv_sha256']
    assert json.loads(json.dumps(parsed, allow_nan=False)) == parsed


def test_filters_and_no_missing_as_zero(table):
    observations = parse_nrel(table)['result']['observations']
    result = filter_observations(observations, enzyme_identifier='E2', temperature_c=40,
                                 ph=6.5, pet_form='CRYSTALLINE_POWDER')
    assert len(result['result']) == 1
    assert result['result'][0]['original_text'] == ' 2.500 '
    assert filter_observations(observations, enzyme_identifier='E3', temperature_c=40,
                               ph=6.5, pet_form='CRYSTALLINE_POWDER')['status'] == 'NO_DATA'
    assert filter_observations(observations, pet_form='AMORPHOUS_FILM')['result'][0]['numerical_value'] == 90


def test_summary_ranks_only_one_exact_condition(table):
    observations = parse_nrel(table)['result']['observations']
    result = summarize_condition(observations, 40, 6.5, 'CRYSTALLINE_POWDER')
    summary = result['result']
    assert result['status'] == 'SUCCESS' and result['evidence_category'] == 'COMPUTED'
    assert summary['observation_count'] == 7
    assert summary['recorded_zeros'] == 1 and summary['positive_recorded_responses'] == 5
    assert [o['enzyme_identifier'] for o in summary['leading_responses']] == ['E4', 'E6', 'E7', 'E8', 'E2']
    assert all(o['pet_form'] == 'CRYSTALLINE_POWDER' for o in summary['leading_responses'])
    assert result['provenance'] == [table['source']]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_no_data_and_empty_observations(table):
    observations = parse_nrel(table)['result']['observations']
    no_data = summarize_condition(observations, 50, 8, 'CRYSTALLINE_POWDER')
    assert no_data['status'] == 'NO_DATA'
    assert no_data['result']['observation_count'] == 0
    assert no_data['result']['leading_responses'] == []
    assert no_data['provenance'] == [table['source']]
    assert summarize_condition([], 50, 8, 'CRYSTALLINE_POWDER')['status'] == 'NO_DATA'


def test_all_missing(table):
    table['rows'] = [table['rows'][0]]
    table['rows'][0][3:] = ['', ' ', '']
    result = parse_nrel(table)
    assert result['status'] == 'NO_DATA'
    assert result['result'] == {'observations': [], 'missing_cells': 3}


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', 'NA', 'invalid', '1e999', '1e-999'])
def test_invalid_values_are_not_zero(table, value):
    table['rows'][0][3] = value
    result = parse_nrel(table)
    assert result['status'] == 'ERROR' and result['result'] is None
    assert value not in (result['message'] or '')


def test_exact_precision_ranking(table):
    table['rows'][0][3] = '0.10000000000000000000001'
    table['rows'][1][3] = '0.1'
    table['rows'] = table['rows'][:2]
    result = summarize_condition(parse_nrel(table)['result']['observations'], 40, 6.5, 'CRYSTALLINE_POWDER')
    assert [o['enzyme_identifier'] for o in result['result']['leading_responses']] == ['E1', 'E2']


@pytest.mark.parametrize('bad_column', ['activity_at_8_50_unknown', 'activity_at_bad_40_cryPow'])
def test_schema_changes_not_silently_ignored(table, bad_column):
    table['columns'][3] = bad_column
    assert parse_nrel(table)['status'] == 'ERROR'


def test_row_width_and_missing_identifier_header(table):
    table['rows'][0].pop()
    assert parse_nrel(table)['status'] == 'ERROR'
    table['columns'][0] = 'invented_id'
    assert select_nrel_protein(table, 'E1')['status'] == 'ERROR'


def test_selection_and_duplicate_ids(table):
    selected = select_nrel_protein(table, 'E2')
    assert selected['result']['amino_acid_sequence'] == 'ACDEG'
    assert selected['result']['source_row'] == 3
    assert select_nrel_protein(table, 'e2')['status'] == 'NO_DATA'
    table['rows'].append(table['rows'][1])
    assert select_nrel_protein(table, 'E2')['status'] == 'ERROR'


@pytest.mark.parametrize('kwargs', [{'pet_form': 'powder'}, {'ph': float('nan')},
                                    {'temperature_c': True}, {'enzyme_identifier': 5}])
def test_bad_filters(table, kwargs):
    observations = parse_nrel(table)['result']['observations']
    assert filter_observations(observations, **kwargs)['status'] == 'ERROR'


def test_load_error_and_empty(tmp_path):
    assert load_nrel(str(tmp_path / 'missing'))['status'] == 'ERROR'
    path = tmp_path / 'empty.csv'
    path.write_text(',sequence,activity_at_6.5_40_cryPow\n')
    assert load_nrel(str(path))['status'] == 'NO_DATA'


def test_real_local_integration():
    path = Path(__file__).resolve().parents[1] / 'data/nrel/label_data.csv'
    if not path.exists():
        pytest.skip('Optional local NREL validation; no download in tests')
    table = load_nrel(str(path))['result']
    assert len(table['rows']) == 213
    parsed = parse_nrel(table)
    assert parsed['status'] == 'SUCCESS'
    assert select_nrel_protein(table, 'DP003')['status'] == 'SUCCESS'
    assert summarize_condition(parsed['result']['observations'], 50, 8, 'CRYSTALLINE_POWDER')['status'] == 'NO_DATA'


def test_hash_matches_exact_bytes_and_repeated_cell_locators(table):
    loaded = load_nrel(table['source']['locator'])['result']
    assert loaded['csv_sha256'] == hashlib.sha256(Path(table['source']['locator']).read_bytes()).hexdigest()
    first = parse_nrel(table)['result']['observations']
    second = parse_nrel(table)['result']['observations']
    assert [(o['source'], o['source_row'], o['source_column']) for o in first] == [
        (o['source'], o['source_row'], o['source_column']) for o in second]


def test_precision_beyond_decimal_context(table):
    table['rows'] = table['rows'][:2]
    table['rows'][0][3] = '0.10000000000000000000000000000001'
    table['rows'][1][3] = '0.10000000000000000000000000000002'
    result = summarize_condition(parse_nrel(table)['result']['observations'], 40, 6.5, 'CRYSTALLINE_POWDER')
    assert [o['enzyme_identifier'] for o in result['result']['leading_responses']] == ['E2', 'E1']


def test_unhashable_source_sequence_keeps_reference(table):
    table['rows'][0][1] = 'ACDX'
    observation = parse_nrel(table)['result']['observations'][0]
    assert observation['sequence_sha256'] is None
    assert observation['sequence_reference'].endswith('#record=2&column=sequence')
    assert observation['original_text'] == '0.00'


def test_empty_id_and_huge_filter_are_structured_errors(table):
    assert select_nrel_protein(table, '')['status'] == 'ERROR'
    observations = parse_nrel(table)['result']['observations']
    assert filter_observations(observations, temperature_c=10 ** 400)['status'] == 'ERROR'
