"""Exact supplied-sequence comparison, without alignment or functional inference."""
from __future__ import annotations

import hashlib

from .contracts import IdentityComparison, NrelProtein, PazyProtein, SequenceFingerprint, ToolResult
from .results import outcome

NORMALIZATION = 'Uppercase and remove whitespace only; SHA-256 of normalized ASCII sequence'
LIMITATIONS = [
    'Exact supplied-sequence identity does not imply equivalent function, activity or assay conditions.',
    'Completeness of source sequences is not independently verified; no alignment is performed.',
    'Empty, ambiguous, gapped, stop-marked or non-residue sequences provide insufficient information.',
    'NREL and PAZy records retain separate identifiers and provenance; no records are merged.',
]


def sequence_fingerprint(sequence: str | None) -> SequenceFingerprint:
    """Fingerprint a sequence by uppercase/whitespace normalization plus SHA-256.

    Purpose: canonical exact-match primitive shared by CSV parsing and
        identity comparison. Accepts the 20 canonical residues plus U and O
        (selenocysteine/pyrrolysine); anything else — ambiguous codes,
        gaps, stops, empty input — yields all-None fields rather than a
        guess. No alignment is performed.
    Args:
        sequence: raw sequence string or None.
    Returns:
        SequenceFingerprint with 'normalized_sequence', 'sha256', and
            'length' (all None when unusable).
    Example:
        >>> sequence_fingerprint(' acdef ')
        {'normalized_sequence': 'ACDEF', 'sha256': ..., 'length': 5}
    Flow: Receives str | None from parse_nrel() or compare_identity(); returns SequenceFingerprint for those functions."""
    if not isinstance(sequence, str):
        return {'normalized_sequence': None, 'sha256': None, 'length': None}
    normalized = ''.join(sequence.split()).upper()
    # Preserve selenocysteine/pyrrolysine; do not claim identity at ambiguous residues.
    if not normalized or not set(normalized) <= set('ACDEFGHIKLMNPQRSTVWYOU'):
        return {'normalized_sequence': None, 'sha256': None, 'length': None}
    return {'normalized_sequence': normalized,
            'sha256': hashlib.sha256(normalized.encode('ascii')).hexdigest(),
            'length': len(normalized)}


def compare_identity(nrel_protein: NrelProtein, pazy_protein: PazyProtein) -> ToolResult[IdentityComparison]:
    """Compare two supplied sequences for exact normalized equality.

    Purpose: the only mechanism that may establish sequence identity
        between an NREL row and a PAZy record. Name or alias equality is
        never sufficient; this full-string comparison is required.
    Args:
        nrel_protein: NrelProtein from select_nrel_protein().
        pazy_protein: transient PazyProtein from get_pazy_metadata().
    Returns:
        ToolResult[IdentityComparison] with 'identity' one of
            EXACT_SEQUENCE_MATCH, DIFFERENT_SEQUENCE, or
            INSUFFICIENT_INFORMATION, plus both identifiers, SHA-256
            hashes, lengths, and the normalization rule.
    Statuses:
        Always SUCCESS (the assessment ran); ambiguity is expressed via
            the INSUFFICIENT_INFORMATION identity value, not an error.
    Example:
        >>> nrel = select_nrel_protein(load_nrel()['result'], 'ESM050')['result']
        >>> out = compare_identity(nrel, {'id': 1, 'amino_acid_sequence': nrel['amino_acid_sequence'], 'source': nrel['source'], **{k: None for k in ('name','uniprot_accession','organism','substrates','literature')}})
        >>> out['result']['identity']
        'EXACT_SEQUENCE_MATCH'
    Limitations:
        Matching sequences do not establish equivalent function, activity,
            or assayed-construct identity; source completeness is unverified.
    Flow: Receives NrelProtein from select_nrel_protein() and PazyProtein from get_pazy_metadata(); returns ToolResult[IdentityComparison] as a final identity assessment."""
    left = sequence_fingerprint(nrel_protein['amino_acid_sequence'])
    right = sequence_fingerprint(pazy_protein['amino_acid_sequence'])
    if left['normalized_sequence'] is None or right['normalized_sequence'] is None:
        identity = 'INSUFFICIENT_INFORMATION'
    elif left['normalized_sequence'] == right['normalized_sequence']:
        identity = 'EXACT_SEQUENCE_MATCH'
    else:
        identity = 'DIFFERENT_SEQUENCE'
    result: IdentityComparison = {
        'identity': identity,
        'nrel_identifier': nrel_protein['enzyme_identifier'],
        'pazy_identifier': pazy_protein['id'],
        'nrel_sequence_sha256': left['sha256'], 'pazy_sequence_sha256': right['sha256'],
        'nrel_sequence_length': left['length'], 'pazy_sequence_length': right['length'],
        'normalization': NORMALIZATION,
    }
    return outcome('SUCCESS', result, [nrel_protein['source'], pazy_protein['source']],
                   LIMITATIONS, 'COMPUTED')
