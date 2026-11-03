"""Conservative preparation of NREL or researcher-supplied local sequences."""
from __future__ import annotations

from .contracts import NrelProtein, Source, ToolResult
from .identity import sequence_fingerprint
from .protein_types import PreparedSequence
from .results import outcome

MAX_RESIDUES = 1022  # Operational bound, leaving two tokens inside a 1,024-token budget.
CANONICAL = frozenset('ACDEFGHIKLMNPQRSTVWY')
AMBIGUOUS = frozenset('BJXZ')
SEQUENCE_LIMITATIONS = [
    'Only the 20 canonical amino acids are embedded; ambiguous/unsupported residues are excluded, not replaced.',
    'Uppercase/whitespace normalization does not verify source completeness or biological identity.',
    'Sequences longer than 1,022 residues are rejected without truncation.',
]


def prepare_protein_sequence(protein: NrelProtein | str | None) -> ToolResult[PreparedSequence]:
    """Validate and normalize a local sequence for embedding (no repair).

    Purpose: accepts an NREL protein or a researcher-supplied string and
        classifies it as VALID, MISSING, AMBIGUOUS_RESIDUES,
        UNSUPPORTED_RESIDUES, or TOO_LONG (>1022 residues). Ambiguous and
        unsupported symbols are flagged, never substituted, and long
        sequences are rejected rather than truncated.
    Args:
        protein: NrelProtein, raw sequence string, or None.
    Returns:
        ToolResult[PreparedSequence] with 'normalized_sequence',
            'sequence_sha256', 'sequence_length', 'validation',
            'flagged_symbols', source row/hash (for NREL inputs), and
            'provenance'.
    Statuses:
        SUCCESS when VALID; NO_DATA when MISSING; ERROR otherwise (the
            result payload is still returned so callers can inspect why).
    Example:
        >>> out = prepare_protein_sequence('ACDE' * 10)
        >>> (out['status'], out['result']['validation'], out['result']['sequence_length'])
        ('SUCCESS', 'VALID', 40)
    Limitations:
        Normalization never verifies biological identity or source
            completeness; researcher input is explicitly unverified.
    Flow: Receives NrelProtein from select_nrel_protein() or external local str | None; returns ToolResult[PreparedSequence] for embed_protein()."""
    local: Source = {'name': 'LOCAL_SEQUENCE', 'url': '', 'doi': None, 'locator': 'researcher-supplied sequence',
                     'attribution': 'Researcher-supplied input; source identity unverified.', 'retrieved_at_utc': None}
    sources = [protein['source']] if isinstance(protein, dict) else [local]
    sequence = protein['amino_acid_sequence'] if isinstance(protein, dict) else protein
    normalized = ''.join(sequence.split()).upper() if isinstance(sequence, str) else ''
    symbols = (set(normalized) - CANONICAL) | {char for char in (sequence if isinstance(sequence,str) else '') if not char.isascii() and not char.isspace()}
    validation = ('MISSING' if not normalized else 'UNSUPPORTED_RESIDUES' if symbols - AMBIGUOUS else
                  'AMBIGUOUS_RESIDUES' if symbols else 'TOO_LONG' if len(normalized) > MAX_RESIDUES else 'VALID')
    fingerprint = sequence_fingerprint(normalized) if validation == 'VALID' else None
    result: PreparedSequence = {
        'identifier': protein['enzyme_identifier'] if isinstance(protein, dict) else None,
        'normalized_sequence': fingerprint['normalized_sequence'] if fingerprint else None,
        'sequence_sha256': fingerprint['sha256'] if fingerprint else None,
        'sequence_length': len(normalized), 'validation': validation,
        'flagged_symbols': sorted(symbols),
        'source_row': protein['source_row'] if isinstance(protein, dict) else None,
        'csv_sha256': protein['csv_sha256'] if isinstance(protein, dict) else None,
        'provenance': sources,
    }
    return outcome('SUCCESS' if validation == 'VALID' else 'NO_DATA' if validation == 'MISSING' else 'ERROR',
                   result, sources, SEQUENCE_LIMITATIONS, 'COMPUTED',
                   None if validation == 'VALID' else 'Sequence not eligible: ' + validation)
