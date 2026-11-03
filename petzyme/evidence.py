"""Deterministic assessment of what supplied observation evidence does not establish."""
from __future__ import annotations

from .assessment_types import (CoverageGap, EvidenceAssessment, EvidenceIssue, MetadataRecord)
from .contracts import Condition, Observation, ToolResult
from .experimental import (LIMITATIONS, PROTOCOL_FIELDS, cell_locator, condition_matches,
                           details_for, metadata_index, provenance, source_key,
                           unique_observations, valid_condition)
from .results import outcome


def assess_evidence(observations: list[Observation], metadata: list[MetadataRecord] | None = None,
                    expected_conditions: list[Condition] | None = None) -> ToolResult[EvidenceAssessment]:
    """Assess what supplied observation evidence does NOT establish.

    Purpose: runs six deterministic issue rules (units, metadata, zeros
        without detection limits, replicates, uncertainty, publication
        mapping) plus coverage-gap detection against an explicit target
        condition set. Repeated references to one cell are deduplicated
        first, so they can never count as replicates.
    Args:
        observations: list from parse_nrel()/filter_observations().
        metadata: optional caller-supplied MetadataRecords, each bound to
            an exact cell locator plus CSV SHA-256 with documentation
            provenance. Undocumented records are ignored (treated as {}).
        expected_conditions: optional explicit target Condition list used
            only to compute missing-condition gaps; omitted means coverage
            completeness cannot be established (INFO issue).
    Returns:
        ToolResult[EvidenceAssessment] with 'observation_count',
            'issues' (code/severity/explanation/affected_cells/provenance),
            'missing_conditions', and 'coverage_scope_explicit'.
    Statuses:
        SUCCESS when observations were supplied; NO_DATA when none were
            (still listing the NO_OBSERVATIONS issue); ERROR on
            conflicting cells or invalid metadata/scope.
    Example:
        >>> obs = parse_nrel(load_nrel()['result'])['result']['observations'][:1]
        >>> out = assess_evidence(obs)
        >>> [i['code'] for i in out['result']['issues']][:2]
        ['UNITS_UNVERIFIED', 'MISSING_EXPERIMENTAL_METADATA']
    Limitations:
        SUCCESS means the assessment ran, not that the evidence is
            sufficient; caller documentation is never authenticated.
    Flow: Receives list[Observation] from filter_observations() and external metadata/coverage scope; returns ToolResult[EvidenceAssessment] for check_comparability() or investigate_conditions()."""
    try:
        selected, index = unique_observations(observations), metadata_index(metadata)
        if expected_conditions is not None and not all(valid_condition(c) for c in expected_conditions):
            raise ValueError('Invalid coverage condition')
    except ValueError:
        return outcome('ERROR', None, provenance(observations, metadata), LIMITATIONS, 'COMPUTED',
                       'Conflicting observations, invalid metadata or invalid coverage scope')
    sources = provenance(selected, metadata)
    issues: list[EvidenceIssue] = []
    rules = [
        ('UNITS_UNVERIFIED', 'HIGH', lambda o, d: not (d.get('units_verified') is True and d.get('measurement_unit')),
         'Measurement units are not verified for these cells; unit equivalence or conversion is unsupported.'),
        ('MISSING_EXPERIMENTAL_METADATA', 'MEDIUM', lambda o, d: not d.get('experimental_group') or not d.get('measurement_definition') or not d.get('protocol_id') or
         any(not d.get('protocol', {}).get(field) for field in PROTOCOL_FIELDS),
         'Measurement definition or cell-linked protocol metadata is incomplete; procedures may have been performed but are not established here.'),
        ('ZERO_WITHOUT_DETECTION_LIMIT', 'HIGH', lambda o, d: o['is_zero'] and not d.get('detection_limit'),
         'A zero was recorded without a documented detection limit in the supplied evidence; this does not establish inactivity.'),
        ('REPLICATES_UNREPORTED', 'MEDIUM', lambda o, d: 'replicates' not in d,
         'Replicate information is not established; repeated source references are not evidence of independent replicates.'),
        ('UNCERTAINTY_UNREPORTED', 'MEDIUM', lambda o, d: not d.get('uncertainty'),
         'Uncertainty estimates or their definitions are not established for the recorded values.'),
        ('PUBLICATION_MAPPING_UNVERIFIED', 'MEDIUM', lambda o, d: not (d.get('publication_mapping_verified') is True and d.get('publication_doi') and
         d.get('publication_doi') != o['source']['doi']),
         'A cell-level publication mapping has not been verified; the dataset DOI alone does not establish that mapping.'),
    ]
    for code, severity, predicate, explanation in rules:
        affected = [observation for observation in selected if predicate(observation, details_for(observation, index))]
        if affected:
            issues.append({'code': code, 'severity': severity, 'explanation': explanation,
                           'affected_cells': [cell_locator(o) for o in affected],
                           'provenance': provenance(affected, metadata)})
    missing: list[CoverageGap] = []
    groups: dict[tuple, list[Observation]] = {}
    for observation in selected:
        groups.setdefault((*source_key(observation), observation['source_row']), []).append(observation)
    for group in groups.values():
        for condition in expected_conditions or []:
            gap: CoverageGap = {'enzyme_identifier': group[0]['enzyme_identifier'], 'condition': condition,
                                'source_row': group[0]['source_row'], 'source': group[0]['source'],
                                'csv_sha256': group[0]['csv_sha256']}
            if not any(condition_matches(o, condition) for o in group) and gap not in missing:
                missing.append(gap)
    if missing:
        issues.append({'code': 'INCOMPLETE_CONDITION_COVERAGE', 'severity': 'HIGH',
                       'explanation': 'Selected records do not cover every explicitly requested condition; no missing response is inferred.',
                       'affected_cells': [cell_locator(o) for o in selected], 'provenance': sources})
    if expected_conditions is None:
        issues.append({'code': 'COVERAGE_SCOPE_UNSPECIFIED', 'severity': 'INFO',
                       'explanation': 'No target condition set was supplied; coverage completeness cannot be established.',
                       'affected_cells': [], 'provenance': sources})
    if not selected:
        issues.append({'code': 'NO_OBSERVATIONS', 'severity': 'HIGH',
                       'explanation': 'No recorded observations were supplied; no experimental conclusion follows.',
                       'affected_cells': [], 'provenance': sources})
    result: EvidenceAssessment = {'observation_count': len(selected), 'issues': issues,
                                  'missing_conditions': missing, 'coverage_scope_explicit': expected_conditions is not None}
    return outcome('SUCCESS' if selected else 'NO_DATA', result, sources, LIMITATIONS, 'COMPUTED')
