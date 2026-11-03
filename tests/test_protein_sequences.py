"""Synthetic sequence preparation fixtures; no real external records."""
import pytest
from petzyme.protein_sequences import MAX_RESIDUES, prepare_protein_sequence
from petzyme.observations import load_nrel, select_nrel_protein


def test_normalize_and_fingerprint():
    left = prepare_protein_sequence(' acd eF\n')
    right = prepare_protein_sequence('ACDEF')
    assert left['status'] == 'SUCCESS'
    assert left['result']['normalized_sequence'] == 'ACDEF'
    assert left['result']['sequence_sha256'] == right['result']['sequence_sha256']
    assert left['evidence_category'] == 'COMPUTED'
    assert left['provenance'][0]['name'] == 'LOCAL_SEQUENCE'


@pytest.mark.parametrize('sequence,validation', [(None,'MISSING'), ('  ','MISSING'),
    ('ACDX','AMBIGUOUS_RESIDUES'), ('BJZ','AMBIGUOUS_RESIDUES'), ('ACDUO','UNSUPPORTED_RESIDUES'),
    ('ACD-*','UNSUPPORTED_RESIDUES'), ('ß','UNSUPPORTED_RESIDUES'), ('>FASTA\nACD','UNSUPPORTED_RESIDUES'),
    ('A' * (MAX_RESIDUES+1),'TOO_LONG')])
def test_invalid_sequences_are_flagged_without_substitution(sequence, validation):
    result = prepare_protein_sequence(sequence)
    assert result['status'] != 'SUCCESS' and result['result']['validation'] == validation
    assert result['result']['normalized_sequence'] is None
    assert result['result']['sequence_sha256'] is None


def test_length_boundary():
    assert prepare_protein_sequence('A' * MAX_RESIDUES)['status'] == 'SUCCESS'


def test_nrel_identity_preserved_and_ambiguous_identifier_rejected(tmp_path):
    path = tmp_path / 'synthetic.csv'
    path.write_text(',sequence\nE1,ACDEF\nE2,ACDX\n')
    table = load_nrel(str(path))['result']
    protein = select_nrel_protein(table, 'E1')['result']
    result = prepare_protein_sequence(protein)
    assert result['result']['identifier'] == 'E1' and result['result']['source_row'] == 2
    assert result['result']['csv_sha256'] == table['csv_sha256']
    assert result['provenance'] == [table['source']]
    assert select_nrel_protein(table, 'MISSING')['status'] == 'NO_DATA'
    table['rows'].append(table['rows'][0])
    assert select_nrel_protein(table, 'E1')['status'] == 'ERROR'
