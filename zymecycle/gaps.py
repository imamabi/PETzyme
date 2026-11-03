"""Ordered verification and measurement requirements, without proposed outcomes."""
from __future__ import annotations

from .assessment_types import ConditionInvestigation, EvidenceGap
from .contracts import ToolResult
from .experimental import LIMITATIONS
from .results import outcome


def identify_evidence_gaps(investigation: ConditionInvestigation) -> ToolResult[list[EvidenceGap]]:
    """Turn an investigation's issues into ordered verification/measurement needs.

    Purpose: maps comparability blockers and evidence issue codes to
        prioritized VERIFICATION gaps (source, units, protocol, protein
        record, detection limit, replication, publication mapping), then
        appends one MEASUREMENT_REQUIREMENT per missing requested
        condition. Priorities are ordinals starting at 1.
    Args:
        investigation: ConditionInvestigation from investigate_conditions().
    Returns:
        ToolResult[list[EvidenceGap]]; each gap has 'priority', 'code',
            'kind', 'requirement' text, and 'target_conditions'.
    Statuses:
        Always SUCCESS (requirements derivation cannot fail on a valid
            investigation object).
    Example:
        >>> inv = investigate_conditions(obs, a, b, enzyme_identifier='ESM050')['result']
        >>> gaps = identify_evidence_gaps(inv)['result']
        >>> gaps[0]['code']
        'VERIFY_DEFINITION_AND_UNITS'
    Limitations:
        Proposes no response values, counts, or outcomes — only what must
            be verified or measured.
    Flow: Receives ConditionInvestigation from investigate_conditions(); returns ToolResult[list[EvidenceGap]] as final ordered verification and measurement requirements."""
    comparison = investigation['comparability']
    issues = {issue['code'] for issue in investigation['evidence']['issues']}
    fields = set(comparison['missing_fields'] + comparison['blocking_differences'])
    # Fixed order below: source identity first, then units, protocol, protein record, and the rest.
    requirements: list[tuple[str, str]] = []
    if 'source' in fields or not investigation['source_provenance']:
        requirements.append(('VERIFY_SOURCE_IDENTITY', 'Verify the exact source record and dataset version before relating measurements.'))
    if 'UNITS_UNVERIFIED' in issues or fields & {'measurement_unit', 'measurement_definition', 'verified_units'}:
        requirements.append(('VERIFY_DEFINITION_AND_UNITS',
                             'Reconcile documented measurement definitions and units at each source cell before comparing magnitudes; do not assume a conversion.'))
    if 'MISSING_EXPERIMENTAL_METADATA' in issues or any(f.startswith('protocol') or f == 'experimental_group' for f in fields):
        requirements.append(('VERIFY_PROTOCOL_AND_GROUP',
                             'Verify a shared experimental group and cell-linked incubation, loadings, buffer, detection and normalization metadata.'))
    if 'protein_record' in fields:
        requirements.append(('VERIFY_MATCHED_PROTEIN_RECORD',
                             'Verify matched protein source records when isolating temperature or pH; names alone cannot establish identity.'))
    if (comparison['observation_count'] < 2 and
            all(entry['status'] == 'OBSERVED' for entry in investigation['conditions'])):
        requirements.append(('CLARIFY_COMPARISON_TARGET',
                             'Clarify the intended independent comparison; repeated requests for the same source cell are not independent measurements.'))
    if 'pet_form' in investigation['changed_variables']:
        requirements.append(('SEPARATE_PET_FORMS',
                             'Keep powder and film results separate; obtain matched-form measurements for any requested same-form comparison.'))
    if len(investigation['changed_variables']) > 1:
        requirements.append(('ISOLATE_REQUESTED_VARIABLE',
                             'Specify the variable of interest and obtain or verify comparisons with other relevant conditions controlled.'))
    if 'ZERO_WITHOUT_DETECTION_LIMIT' in issues:
        requirements.append(('VERIFY_DETECTION_LIMIT',
                             'Verify the assay detection limit and interpretation of reported zeros; do not infer inactivity.'))
    if issues & {'REPLICATES_UNREPORTED', 'UNCERTAINTY_UNREPORTED'}:
        requirements.append(('VERIFY_REPLICATION_AND_UNCERTAINTY',
                             'Verify independent replicate information and uncertainty definitions; if unavailable, require their measurement or estimation without inventing counts.'))
    if 'PUBLICATION_MAPPING_UNVERIFIED' in issues:
        requirements.append(('VERIFY_PUBLICATION_MAPPING',
                             'Verify the publication-to-source-cell relationship; a dataset DOI or matching protein sequence is insufficient.'))
    gaps: list[EvidenceGap] = [
        {'priority': index, 'code': code, 'kind': 'VERIFICATION', 'requirement': text, 'target_conditions': []}
        for index, (code, text) in enumerate(requirements, start=1)]
    missing = []
    for entry in investigation['conditions']:
        if entry['status'] == 'MISSING' and entry['condition'] not in missing:
            missing.append(entry['condition'])
    for condition in missing:
        gaps.append({'priority': len(gaps) + 1, 'code': 'MEASURE_REQUESTED_CONDITION',
                     'kind': 'MEASUREMENT_REQUIREMENT',
                     'requirement': 'Obtain an exact-condition response with verified definition, units, matched protocol and documented replication/uncertainty; no response value is proposed.',
                     'target_conditions': [condition]})
    return outcome('SUCCESS', gaps, investigation['source_provenance'], LIMITATIONS, 'COMPUTED')
