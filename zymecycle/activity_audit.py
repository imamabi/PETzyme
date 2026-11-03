"""Deterministic inventories over curated activity records; no pooling or measurement conversion."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

from .activity_snapshot import LIMITATIONS, iter_assays
from .activity_types import ActivitySnapshot
from .contracts import JSONValue, Source, ToolResult
from .results import finite_number, outcome

ASSAY_FIELDS = [
    'conditions.temperature_c', 'conditions.ph', 'conditions.duration_value', 'conditions.duration_unit',
    'conditions.enzyme_amount_value', 'conditions.enzyme_amount_unit',
    'conditions.enzyme_concentration_value', 'conditions.enzyme_concentration_unit',
    'conditions.substrate_amount_value', 'conditions.substrate_amount_unit',
    'conditions.substrate_concentration_value', 'conditions.substrate_concentration_unit',
    'conditions.reaction_volume_value', 'conditions.reaction_volume_unit', 'conditions.buffer',
    'material.form', 'material.morphology', 'material.crystallinity_percent', 'material.pretreatment',
    'material.pretreatment_steps', 'detection_methods', 'evidence.source', 'evidence.location',
]
MEASUREMENT_FIELDS = ['value', 'unit', 'metric_type', 'analyte', 'qualifier', 'categorical_result',
                      'reported_result', 'replicate_n', 'replicate_n_reported', 'uncertainty_value',
                      'uncertainty_type', 'value_is_digitized']


def field_coverage(records: list[dict], path: str) -> dict[str, int]:
    """Count one dotted field path as documented/absent/null/empty across records.

    Purpose: distinguishes "field missing" from "present but null" from
        "present but empty" so coverage summaries never conflate absent
        documentation with negative evidence.
    Flow: Receives source dictionaries and field path str from audit_pazy_pet_assays(); returns dict[str, int] separating documented, absent, null and empty fields for its coverage summary."""
    counts = Counter({'documented': 0, 'absent': 0, 'null': 0, 'empty': 0})
    for record in records:
        current = record
        for key in path.split('.'):
            if not isinstance(current, dict) or key not in current:
                counts['absent'] += 1
                break
            current = current[key]
        else:
            counts['null' if current is None else 'empty' if current in ('', [], {}) else 'documented'] += 1
    return dict(counts)


def audit_pazy_pet_assays(snapshot: ActivitySnapshot) -> ToolResult[dict[str, JSONValue]]:
    """Inventory the snapshot's PET assays: counts, duplicates, field coverage.

    Purpose: Build 07A audit composition. Reports assay/measurement
        counts, repeated-assay fingerprints (same protein/DOI/content),
        result-kind and digitization tallies, per-field coverage, and
        value inventories. Pure counting — no pooling, conversion, or
        measurement interpretation.
    Returns:
        ToolResult[dict] of aggregate inventory/coverage counts.
    Flow: Receives ActivitySnapshot from load_pazy_activity_snapshot(); returns ToolResult[dict[str, JSONValue]] with computed inventory and coverage for audit_build07a.main()."""
    selected = list(iter_assays(snapshot))
    assays = [r['assay'] for r in selected]
    measurements = [m for a in assays for m in a['measurements']]
    all_assays = list(iter_assays(snapshot, None))
    parents = Counter((r['protein_id'], r['literature_doi'].casefold()) for r in snapshot['records'])
    repeated_assays = Counter()
    for ref in all_assays:
        assay = {k: v for k, v in ref['assay'].items() if k not in ('is_representative', 'representative_group')}
        fingerprint = hashlib.sha256(json.dumps(assay, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        repeated_assays[(ref['protein_id'], ref['publication_doi'].casefold(), fingerprint)] += 1
    kinds = Counter({'numeric': 0, 'categorical_without_numeric': 0, 'text_only': 0, 'missing_result': 0})
    digitized = Counter({'true': 0, 'false': 0, 'unavailable': 0})
    for m in measurements:
        kind = ('numeric' if finite_number(m.get('value')) else 'categorical_without_numeric' if m.get('categorical_result')
                else 'text_only' if m.get('reported_result') else 'missing_result')
        kinds[kind] += 1
        digitized['true' if m.get('value_is_digitized') is True else 'false' if m.get('value_is_digitized') is False else 'unavailable'] += 1
    def inventory(records: list[dict], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(r[key]) if r.get(key) is not None else '<unavailable>' for r in records).items()))
    result: dict[str, JSONValue] = {
        'snapshot_counts': snapshot['counts'], 'pet_parents': len({r['parent_index'] for r in selected}),
        'pet_assays': len(assays), 'pet_representative_assays': sum(a['is_representative'] for a in assays),
        'pet_measurements': len(measurements), 'pet_proteins': len({r['protein_id'] for r in selected}),
        'pet_publications': len({r['publication_doi'].casefold() for r in selected}),
        'all_substrate_assay_counts': dict(Counter(r['assay']['substrate'] for r in all_assays)),
        'measurement_kinds': dict(kinds), 'digitization': dict(digitized),
        'numeric_zero_measurements': sum(finite_number(m.get('value')) and m['value'] == 0 for m in measurements),
        'categorical_field_present': sum(bool(m.get('categorical_result')) for m in measurements),
        'numeric_with_qualifier': sum(finite_number(m.get('value')) and bool(m.get('qualifier')) for m in measurements),
        'numeric_replicate_counts': sum(type(m.get('replicate_n')) is int for m in measurements),
        'textual_replicate_counts': sum(isinstance(m.get('replicate_n'), str) for m in measurements),
        'assay_field_coverage': {field: field_coverage(assays, field) for field in ASSAY_FIELDS},
        'measurement_field_coverage': {field: field_coverage(measurements, field) for field in MEASUREMENT_FIELDS},
        'metric_inventory': inventory(measurements, 'metric_type'), 'unit_inventory': inventory(measurements, 'unit'),
        'analyte_inventory': inventory(measurements, 'analyte'), 'qualifier_inventory': inventory(measurements, 'qualifier'),
        'form_inventory': inventory([a['material'] for a in assays], 'form'),
        'morphology_inventory': inventory([a['material'] for a in assays], 'morphology'),
        'duration_unit_inventory': inventory([a['conditions'] for a in assays], 'duration_unit'),
        'method_inventory': dict(Counter(m['method'] for a in assays for m in a['detection_methods'] if m.get('method'))),
        'duplicate_parent_key_groups': sum(n > 1 for n in parents.values()),
        'exact_curation_duplicate_candidate_groups': sum(n > 1 for n in repeated_assays.values()),
        'exact_curation_duplicate_candidate_excess_assays': sum(n - 1 for n in repeated_assays.values() if n > 1),
    }
    return outcome('SUCCESS' if assays else 'NO_DATA', result, [snapshot['source']], [*LIMITATIONS,
        'Coverage denominators are PET assays or individual measurements as labeled; arrays of methods are not assay counts.',
        'Duplicate curation fingerprints are candidates for review, not proof of independent or duplicate original experiments.',
        'Original unit strings, qualifiers, nulls and textual replicate descriptions are preserved; no equivalence inferred.'], 'COMPUTED')


def audit_publication_workbook_links(snapshot: ActivitySnapshot, workbook_rows: dict[int, dict[str, JSONValue]],
                                     workbook_source: Source) -> ToolResult[dict[str, JSONValue]]:
    """Flow: Receives ActivitySnapshot from load_pazy_activity_snapshot() and external in-memory NREL Activity rows/source; returns ToolResult[dict[str, JSONValue]] as aggregate source-link validation, never a dataset copy."""
    counts = Counter({'assays': 0, 'resolved_row_reference_assays': 0, 'matching_protein_condition_form_assays': 0,
                      'matching_activity_value_groups': 0, 'matching_conversion_value_groups': 0,
                      'activity_value_group_mismatches': 0, 'conversion_value_group_mismatches': 0,
                      'activity_measurements_checked': 0, 'conversion_measurements_checked': 0})
    referenced = Counter()
    for reference in iter_assays(snapshot):
        if reference['publication_doi'].casefold() != '10.1021/acscatal.5c03460': continue
        counts['assays'] += 1
        assay = reference['assay']
        location = assay['evidence'].get('location', '')
        match = re.search(r'cs5c03460_si_002\.xlsx, Activity sheet, rows? ([0-9, ]+)', location)
        indices = [int(n) for n in re.findall(r'\d+', match.group(1))] if match else []
        if not indices or any(i not in workbook_rows for i in indices): continue
        counts['resolved_row_reference_assays'] += 1
        referenced.update(indices)
        rows = [workbook_rows[i] for i in indices]
        conditions = assay['conditions']
        form = {'powder': 'cryPow', 'film': 'aFilm'}.get(assay['material'].get('form'))
        structural = all(r.get('sample_id') == reference['protein_name'] and r.get('pH') == conditions.get('ph')
                         and r.get('temperature_c') == conditions.get('temperature_c') and r.get('substrate') == form for r in rows)
        counts['matching_protein_condition_form_assays'] += int(structural)
        for metric, column, prefix in [('specific_activity','umol_product_per_mg_enzyme','activity'),
                                        ('conversion_percent','percent_depolymerization','conversion')]:
            values = [m.get('value') for m in assay['measurements'] if m.get('metric_type') == metric]
            expected = [r.get(column) for r in rows]
            counts[prefix + '_measurements_checked'] += len(values)
            same = (len(values) == len(expected) and all(finite_number(v) for v in values + expected)
                    and all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9) for a,b in zip(sorted(values),sorted(expected))))
            counts['matching_' + prefix + '_value_groups' if same else prefix + '_value_group_mismatches'] += 1
    return outcome('SUCCESS', {**dict(counts), 'unique_referenced_workbook_rows': len(referenced),
        'reused_workbook_row_indices': sum(n > 1 for n in referenced.values()),
        'workbook_row_reference_occurrences': sum(referenced.values())},
        [snapshot['source'], workbook_source], [*LIMITATIONS,
        'Exact cited workbook-row/value agreement verifies curation links to that workbook, not authenticated CSV aggregation or independent new experiments.',
        'Group comparison ignores measurement order but preserves multiplicity; arithmetic tolerance is not assay uncertainty.',
        'This does not assign a unit to any original NREL CSV cell or merge source measurements.'], 'COMPUTED')
