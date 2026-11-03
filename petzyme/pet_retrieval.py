"""Bounded, read-only PET evidence selection from the authorized activity archive."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import TypedDict

from .activity_audit import ASSAY_FIELDS, MEASUREMENT_FIELDS, field_coverage
from .activity_snapshot import FILENAME, SHA256, LIMITATIONS, iter_assays, load_pazy_activity_snapshot
from .activity_types import ActivitySnapshot, AssayReference
from .contracts import JSONValue, ToolResult
from .results import finite_number, outcome

ROOT = Path(__file__).resolve().parents[1]
MAX_ASSAYS = 3
MAX_MEASUREMENTS = 3


class EvidenceQuery(TypedDict, total=False):
    parent_index: int
    assay_index: int
    measurement_index: int
    protein_id: int
    protein_name: str
    publication_doi: str
    temperature_c: float
    ph: float
    duration_value: float | str
    duration_unit: str
    form: str
    morphology: str
    crystallinity_percent: float
    buffer: str
    pretreatment: str
    detection_method: str
    enzyme_amount_value: float
    enzyme_amount_unit: str
    substrate_amount_value: float
    substrate_amount_unit: str
    reaction_volume_value: float
    reaction_volume_unit: str


def load_activity_source() -> ToolResult[ActivitySnapshot]:
    """Load the pinned PAZy activity archive after checking its manifest pin.

    Purpose: the only sanctioned entry to publication-derived PAZy assay
        data. Verifies filename, published/verified SHA-256, and local
        path against PAZY_SOURCE_MANIFEST.json before loading, then stamps
        the snapshot source with the manifest retrieval time.
    Returns:
        ToolResult[ActivitySnapshot]; ERROR with no fallback when the
            manifest is missing or inconsistent.
    Flow: Receives the external PAZY_SOURCE_MANIFEST.json pin; returns load_pazy_activity_snapshot() ToolResult[ActivitySnapshot] for investigate_pet_evidence()."""
    try:
        manifest = json.loads((ROOT / 'PAZY_SOURCE_MANIFEST.json').read_text())
        if (manifest['filename'] != FILENAME or manifest['published_sha256'] != SHA256
                or manifest['verified_sha256'] != SHA256 or manifest['local_archive'] != 'data/pazy/' + FILENAME):
            raise ValueError('Source pin changed')
        retrieved_at = manifest['retrieved_at_utc']
        if not isinstance(retrieved_at,str): raise ValueError('Missing retrieval provenance')
    except (OSError, ValueError, KeyError, TypeError):
        return outcome('ERROR', None, [], LIMITATIONS, None, 'Authorized activity manifest unavailable or inconsistent; no fallback.')
    result = load_pazy_activity_snapshot(str(ROOT / manifest['local_archive']))
    if result['result'] is not None:
        result['result']['source']['retrieved_at_utc'] = retrieved_at
    return result


def valid_evidence_query(query: EvidenceQuery) -> bool:
    """Validate an EvidenceQuery: known keys only, typed values, no coercion.

    Purpose: rejects unknown filter keys (typos cannot silently narrow a
        search), enforces index/numeric/string shapes per key, and caps
        strings at 120 chars. Runs before any archive access.
    Returns:
        bool; True only for a fully well-formed query dict.
    Flow: Receives external EvidenceQuery from retrieve_pet_assays() or investigate_pet_evidence(); returns bool for exact-filter validation before source access."""
    if not isinstance(query, dict) or not query.keys() <= EvidenceQuery.__annotations__.keys():
        return False
    indexes = {'parent_index', 'assay_index', 'measurement_index', 'protein_id'}
    numeric = {'temperature_c', 'ph', 'crystallinity_percent', 'enzyme_amount_value',
               'substrate_amount_value', 'reaction_volume_value'}
    for key, value in query.items():
        if key in indexes:
            if type(value) is not int or value < (1 if key == 'protein_id' else 0): return False
        elif key in numeric:
            if not finite_number(value): return False
        elif key == 'duration_value' and finite_number(value):
            continue
        elif not isinstance(value, str) or not value.strip() or len(value) > 120:
            return False
    return True


def match_pet_assays(snapshot: ActivitySnapshot, query: EvidenceQuery) -> list[AssayReference]:
    """Find every PET assay in the snapshot matching all exact filters.

    Purpose: conjunctive exact matching over protein names/aliases,
        publication DOI (case-insensitive), positional indexes, and
        condition/material fields. No ranking, no fuzzy matching.
    Args:
        snapshot: ActivitySnapshot from load_activity_source().
        query: validated EvidenceQuery (invalid queries raise).
    Returns:
        list[AssayReference] in source order; empty when nothing matches.
    Raises:
        ValueError: on an invalid query.
    Flow: Receives ActivitySnapshot from load_activity_source() and validated EvidenceQuery; returns list[AssayReference] for retrieve_pet_assays() and investigate_pet_evidence() aggregate coverage."""
    if not valid_evidence_query(query): raise ValueError('Invalid exact evidence filters')
    matches = []
    for ref in iter_assays(snapshot):
        record = snapshot['records'][ref['parent_index']]
        assay = ref['assay']
        # Documented aliases count as name hits; matching still never upgrades to verified identity.
        names = [record['protein_name'], *(s.get('name') for s in record.get('synonyms') or [])]
        matched = True
        for key, value in query.items():
            if key == 'protein_name': actual = value if value in names else None
            elif key == 'publication_doi': actual, value = ref[key].casefold(), value.casefold()
            elif key in ('parent_index', 'assay_index', 'protein_id'): actual = ref[key]
            elif key == 'measurement_index': actual = value if value < len(assay['measurements']) else None
            elif key == 'detection_method': actual = value if any(m.get('method') == value for m in assay['detection_methods']) else None
            elif key in ('form', 'morphology', 'crystallinity_percent', 'pretreatment'): actual = assay['material'].get(key)
            else: actual = assay['conditions'].get(key)
            if actual != value:
                matched = False
                break
        if matched: matches.append(ref)
    return matches


def diverse_assays(matches: list[AssayReference], limit: int) -> list[AssayReference]:
    """Select up to `limit` assays preferring publication, then condition, diversity.

    Purpose: three passes (distinct publication → distinct
        publication/temperature/pH/form → source order) fill the bounded
        preview with varied evidence. Activity values never influence
        selection, so cherry-picking high responses is impossible.
    Args:
        matches: full match list from match_pet_assays().
        limit: 1..3 assays to select.
    Returns:
        Bounded list[AssayReference].
    Raises:
        ValueError: when limit is outside 1..3.
    Flow: Receives list[AssayReference] from match_pet_assays(); returns bounded list[AssayReference] for project_pet_assays() or investigate_pet_evidence(), prioritizing publication then condition diversity without activity ranking."""
    if type(limit) is not int or not 1 <= limit <= MAX_ASSAYS: raise ValueError('Use a limit of 1–3 assays')
    selected, seen = [], set()
    for key in ('publication', 'condition', 'source_order'):
        for ref in matches:
            identity = (ref['parent_index'], ref['assay_index'])
            if identity in {(r['parent_index'], r['assay_index']) for r in selected}: continue
            assay = ref['assay']
            group = (ref['publication_doi'].casefold() if key == 'publication' else
                     (ref['publication_doi'].casefold(), assay['conditions'].get('temperature_c'),
                      assay['conditions'].get('ph'), assay['material'].get('form')) if key == 'condition' else identity)
            if (key, group) in seen: continue
            seen.add((key, group)); selected.append(ref)
            if len(selected) == limit: return selected
    return selected


def retrieve_pet_assays(snapshot: ActivitySnapshot, query: EvidenceQuery | None = None,
                        limit: int = MAX_ASSAYS) -> ToolResult[dict[str, JSONValue]]:
    """Match and project PET assays for a query (match + project in one call).

    Purpose: convenience composition used by investigate_pet_evidence:
        validates the query and limit, finds all matches, then returns
        the bounded diverse projection with original values and
        positional JSON-pointer locators preserved.
    Args:
        snapshot: ActivitySnapshot from load_activity_source().
        query: EvidenceQuery or None (None means match everything).
        limit: 1..3 assays in the returned preview.
    Returns:
        ToolResult[dict] with 'records', match/return counts,
            truncation flags, and selection policy (see project_pet_assays).
    Statuses:
        SUCCESS/PARTIAL/NO_DATA mirror the projection; ERROR on invalid
            query or limit.
    Flow: Receives ActivitySnapshot from load_activity_source() and external EvidenceQuery/limit; returns bounded ToolResult[dict[str, JSONValue]] for investigate_pet_evidence(), preserving original values and positional source identifiers."""
    query = {} if query is None else query
    if not valid_evidence_query(query) or type(limit) is not int or not 1 <= limit <= MAX_ASSAYS:
        return outcome('ERROR', None, [], LIMITATIONS, 'COMPUTED', 'Use exact documented filters and a limit of 1–3 assays.')
    matches = match_pet_assays(snapshot, query)
    return project_pet_assays(snapshot, matches, query, limit)


def project_pet_assays(snapshot: ActivitySnapshot, matches: list[AssayReference], query: EvidenceQuery,
                       limit: int) -> ToolResult[dict[str, JSONValue]]:
    """Project the bounded diverse preview cards for matched assays.

    Purpose: builds transient per-assay cards (conditions, material,
        detection, evidence location, at most 3 measurements each) with
        original values deep-copied and omissions counted explicitly.
        The v4.2 archive has no native assay IDs, so zero-based JSON
        pointers identify original positions within the pinned file.
    Args:
        snapshot: ActivitySnapshot from load_activity_source().
        matches: full match list from match_pet_assays().
        query: the validated query (a measurement_index filter selects
            exactly that measurement per assay).
        limit: 1..3 assays in the preview.
    Returns:
        ToolResult[dict] with 'records', 'total_matches', 'returned',
            'truncated', archive identifiers, match diversity counts,
            'name_ambiguity', and the 'selection' policy string.
    Statuses:
        SUCCESS when nothing was omitted; PARTIAL when assays or
            measurements were truncated; NO_DATA on zero matches; ERROR
            on invalid query/limit.
    Limitations:
        Preview cards are transient: never saved, exported, or cached.
    Flow: Receives validated list[AssayReference]/EvidenceQuery from retrieve_pet_assays() or investigate_pet_evidence(); returns bounded ToolResult[dict[str, JSONValue]] for their final evidence selection."""
    if not valid_evidence_query(query) or type(limit) is not int or not 1 <= limit <= MAX_ASSAYS:
        return outcome('ERROR', None, [], LIMITATIONS, 'COMPUTED', 'Use a limit of 1–3 assays and exact filters.')
    cards = []
    for ref in diverse_assays(matches, limit):
        assay = ref['assay']
        indices = [query['measurement_index']] if 'measurement_index' in query else list(range(min(MAX_MEASUREMENTS, len(assay['measurements']))))
        missing = [field for field in ASSAY_FIELDS if field_coverage([assay], field)['documented'] == 0]
        measurements = []
        pointer = f"#/{ref['parent_index']}/assays/{ref['assay_index']}"
        for index in indices:
            measurement = assay['measurements'][index]
            measurements.append({'measurement_index': index, 'locator': pointer + f'/measurements/{index}',
                                 'record': deepcopy(measurement), 'evidence_category': 'MEASURED',
                                 'evidence_origin': 'PUBLICATION_DERIVED_CURATION',
                                 'missing_fields': [f for f in MEASUREMENT_FIELDS if field_coverage([measurement], f)['documented'] == 0]})
        cards.append({k: ref[k] for k in ('parent_index', 'assay_index', 'protein_id', 'protein_name', 'publication_doi')})
        cards[-1].update(locator=pointer, evidence_locator=assay['evidence'].get('location'),
                         context=deepcopy({k: assay[k] for k in ('conditions', 'material', 'detection_methods', 'evidence')}),
                         measurements=measurements, total_measurements=len(assay['measurements']),
                         omitted_measurements=0 if 'measurement_index' in query else len(assay['measurements'])-len(measurements),
                         out_of_scope_measurements=len(assay['measurements'])-1 if 'measurement_index' in query else 0, missing_fields=missing,
                         identity_status='UNVERIFIED', source=deepcopy(ref['source']))
    truncated = len(matches) > len(cards) or any(c['omitted_measurements'] for c in cards)
    result = {'records': cards, 'total_matches': len(matches), 'returned': len(cards), 'truncated': truncated,
              'archive_sha256':snapshot['sha256'], 'schema_version':snapshot['schema_version'],
              'total_matching_measurements': sum(1 if 'measurement_index' in query else len(r['assay']['measurements']) for r in matches),
              'matching_parent_records': len({r['parent_index'] for r in matches}),
              'matching_protein_ids': len({r['protein_id'] for r in matches}),
              'matching_publications': len({r['publication_doi'].casefold() for r in matches}),
              'name_ambiguity': 'protein_name' in query and len({r['protein_id'] for r in matches}) > 1,
              'selection': 'First assay per publication, then distinct temperature/pH/form, then source order; no activity ranking.',
              'native_identifier_policy': 'No native parent/assay/measurement ID is supplied by v4.2; zero-based JSON pointers identify original positions within the checksum-pinned archive.'}
    return outcome('PARTIAL' if truncated else 'SUCCESS' if cards else 'NO_DATA', result, [snapshot['source']],
                   [*LIMITATIONS, 'Names/aliases are unresolved candidates, not verified identity. Exact filters do not convert units or fill missing fields.',
                    'At most three assays and three measurements per assay; omissions are explicit. Records are transient and must not be saved or exported.'], 'COMPUTED')
