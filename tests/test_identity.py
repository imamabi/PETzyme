import hashlib
import json

import pytest

from petzyme.identity import compare_identity, sequence_fingerprint
from petzyme.results import outcome

SOURCE = {'name': 'synthetic', 'url': 'https://example.invalid/', 'doi': None,
          'locator': 'synthetic.csv#record=2', 'attribution': 'Synthetic fixture', 'retrieved_at_utc': None}


def proteins(left: str | None, right: str | None) -> tuple[dict, dict]:
    return ({'enzyme_identifier': 'same name', 'amino_acid_sequence': left,
             'source_row': 2, 'source': SOURCE, 'csv_sha256': 'synthetic'},
            {'id': 1, 'name': 'same name', 'amino_acid_sequence': right,
             'uniprot_accession': None, 'organism': None, 'substrates': None, 'literature': None,
             'source': {**SOURCE, 'name': 'synthetic PAZy'}})


@pytest.mark.parametrize('left,right,expected', [
    (' acd\nEF ', 'ACDEF', 'EXACT_SEQUENCE_MATCH'),
    ('ACDEF', 'ACDEG', 'DIFFERENT_SEQUENCE'),
    ('ACDE', 'ACDEF', 'DIFFERENT_SEQUENCE'),
    (None, 'ACDEF', 'INSUFFICIENT_INFORMATION'),
    ('', '', 'INSUFFICIENT_INFORMATION'),
    ('ACD-EF', 'ACDEF', 'INSUFFICIENT_INFORMATION'),
    ('ACD*', 'ACD', 'INSUFFICIENT_INFORMATION'),
    ('ACDX', 'ACDX', 'INSUFFICIENT_INFORMATION'),
    ('ACD...', 'ACD', 'INSUFFICIENT_INFORMATION'),
])
def test_identity(left, right, expected):
    nrel, pazy = proteins(left, right)
    before = json.dumps([nrel, pazy])
    result = compare_identity(nrel, pazy)
    assert result['result']['identity'] == expected
    assert result['evidence_category'] == 'COMPUTED'
    assert result['provenance'] == [nrel['source'], pazy['source']]
    assert json.dumps([nrel, pazy]) == before
    json.dumps(result, allow_nan=False)
    if expected == 'EXACT_SEQUENCE_MATCH':
        assert result['result']['nrel_sequence_sha256'] == hashlib.sha256(b'ACDEF').hexdigest()


def test_fingerprint_is_deterministic():
    assert sequence_fingerprint('acdef') == sequence_fingerprint('A C D E F')
    assert sequence_fingerprint('uo')['length'] == 2


def test_envelope():
    limitations = ['no predictions']
    result = outcome('NO_DATA', None, [], limitations, None, 'No records')
    assert result['status'] == 'NO_DATA' and result['message'] == 'No records'
    limitations.append('changed')
    assert result['limitations'] == ['no predictions']
    json.dumps(result, allow_nan=False)
