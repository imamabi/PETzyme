"""Read one checksum-pinned activity archive; no downloads, extraction or protein API cache."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import zlib
from pathlib import Path
from typing import Iterator

from .activity_types import ActivitySnapshot, AssayReference
from .contracts import Source, ToolResult
from .results import outcome

FILENAME = 'pazy_polymer_assays_2026-10-27_113456_345688Z_6cd009d5.json.gz'
SHA256 = '123f058a6a5b34acbba0c1bc7f69d924098045cb9361141a13462c366295c363'
URL = 'https://www.pazy.eu/downloads/archive/' + FILENAME
MAX_COMPRESSED = 4_000_000
MAX_JSON = 64_000_000
LIMITATIONS = ['PAZy assay entries are publication-derived curation, not new independent experiments.',
               'Missing metadata is not proof that a procedure was absent; no source values or units are converted.',
               'Only the pinned activity snapshot may be stored; live protein/API and sequence results remain transient.']

# Fields from the official v4.2 empty template. Optional fields may be absent or null.
FIELDS = {
    'record': set('schema_version protein_id protein_name synonyms literature_doi reference_role supporting_types assays'.split()),
    'synonym': set('name synonym_type source_doi source_database'.split()),
    'assay': set('substrate is_representative representative_group material conditions degradation_chemistry detection_methods measurements evidence'.split()),
    'material': set('reported_name material_category polymer_architecture degree_of_polymerization degree_of_polymerization_reported form morphology crystallinity_percent mw_value mw_unit mn_value mn_unit dispersity_mw_mn molecular_weight_reported supplier thickness_value thickness_unit particle_size_value particle_size_unit surface_area_value surface_area_unit dimensions pretreatment pretreatment_steps'.split()),
    'step': set('step_order type description reagent concentration_value concentration_unit temperature_c duration_value duration_min_value duration_max_value duration_unit duration_reported'.split()),
    'conditions': set('temperature_c temperature_reported ph duration_value duration_unit enzyme_concentration_value enzyme_concentration_unit enzyme_amount_value enzyme_amount_unit enzyme_preparation enzyme_features substrate_amount_value substrate_amount_unit substrate_concentration_value substrate_concentration_unit reaction_volume_value reaction_volume_unit agitation_mode agitation_speed_rpm buffer additional_conditions'.split()),
    'chemistry': set('reaction_type bond_type evidence_level reported_description evidence_source evidence_location'.split()),
    'method': {'method', 'role'},
    'measurement': set('metric_type analyte value unit qualifier categorical_result reported_result uncertainty_value uncertainty_type replicate_n replicate_n_reported value_is_digitized'.split()),
    'evidence': {'source', 'location'},
}
NUMERIC = set('degree_of_polymerization crystallinity_percent mw_value mn_value dispersity_mw_mn thickness_value particle_size_value surface_area_value concentration_value temperature_c duration_min_value duration_max_value ph enzyme_concentration_value enzyme_amount_value substrate_amount_value substrate_concentration_value reaction_volume_value agitation_speed_rpm value uncertainty_value'.split())


def _object(value: object, kind: str) -> None:
    """Validate actual fixed-schema shapes without filling defaults or changing nulls."""
    if not isinstance(value, dict) or not set(value) <= FIELDS[kind]:
        raise ValueError('Unexpected activity schema fields')
    children = {'synonyms': ('synonym', True), 'assays': ('assay', True), 'material': ('material', False),
                'conditions': ('conditions', False), 'pretreatment_steps': ('step', True),
                'degradation_chemistry': ('chemistry', True), 'detection_methods': ('method', True),
                'measurements': ('measurement', True), 'evidence': ('evidence', False)}
    for key, item in value.items():
        if item is None:
            continue
        if key in children:
            child, many = children[key]
            if many:
                if not isinstance(item, list): raise ValueError('Expected list')
                for entry in item: _object(entry, child)
            else: _object(item, child)
        elif key == 'additional_conditions':
            if not isinstance(item, dict): raise ValueError('Expected additional conditions object')
        elif key in ('supporting_types', 'enzyme_features'):
            if not isinstance(item, list) or any(not isinstance(x, str) for x in item): raise ValueError('Expected text list')
        elif key in ('is_representative', 'value_is_digitized'):
            if type(item) is not bool: raise ValueError('Expected boolean')
        elif key in ('protein_id', 'step_order'):
            if type(item) is not int or item < 1: raise ValueError('Expected positive identifier/order')
        elif key in ('replicate_n', 'duration_value'):
            # Source v4.2 also retains textual ranges/replicate descriptions.
            if type(item) not in (int, float, str): raise ValueError('Expected reported number or text')
        elif key in NUMERIC:
            if type(item) not in (int, float) or not math.isfinite(item): raise ValueError('Expected finite number')
        elif not isinstance(item, str):
            raise ValueError('Expected source text')
    if kind == 'record':
        if (value.get('schema_version') != '4.2' or type(value.get('protein_id')) is not int
                or not value.get('protein_name') or not isinstance(value.get('literature_doi'), str)
                or value['literature_doi'].startswith('10.0000/') or not isinstance(value.get('assays'), list)):
            raise ValueError('Missing production record identity/schema')
    if kind == 'assay':
        required = {'substrate': str, 'is_representative': bool, 'material': dict, 'conditions': dict,
                    'measurements': list, 'detection_methods': list, 'evidence': dict}
        if any(type(value.get(k)) is not t for k, t in required.items()): raise ValueError('Missing assay structure')


def _pairs(pairs: list[tuple]) -> dict:
    """Reject duplicate keys rather than silently overwriting source evidence."""
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _float(text: str) -> float:
    """Reject overflowing JSON numbers at every nesting level."""
    value = float(text)
    if not math.isfinite(value): raise ValueError('Nonfinite JSON number')
    return value


def load_pazy_activity_snapshot(path: str = 'data/pazy/' + FILENAME, *, expected_sha256: str = SHA256,
                                expected_counts: tuple[int, int, int] | None = (639, 7740, 645)) -> ToolResult[ActivitySnapshot]:
    """Load the pinned PAZy activity archive with checksum, bounds, and count checks.

    Purpose: verifies the gzip file's SHA-256, caps compressed and
        expanded sizes, parses JSON with duplicate-key rejection and
        finite-float enforcement, validates every record's schema, and
        confirms the published (parents, assays, representative) counts.
        Any failure yields ERROR with no partial data.
    Args:
        path: local archive path (default: the pinned data/pazy file).
        expected_sha256: required file digest.
        expected_counts: required (parents, assays, representative)
            tuple, or None to skip the count check.
    Returns:
        ToolResult[ActivitySnapshot] with records, schema_version,
            sha256, counts, and source. Evidence category None.
    Flow: Receives an external local archive path str and integrity expectations; returns ToolResult[ActivitySnapshot] for audit_pazy_pet_assays() and identify_cross_source_overlap()."""
    source: Source = {'name': 'PAZy activity snapshot', 'url': URL, 'doi': None,
                      'locator': path, 'attribution': 'PAZy / Feuerriegel et al. (2026), database article doi:10.1093/nar/gkag1026; publication-derived curation; dataset license not established.',
                      'retrieved_at_utc': None}
    try:
        if Path(path).stat().st_size > MAX_COMPRESSED: raise ValueError('Archive size bound')
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256: raise ValueError('Checksum mismatch')
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as archive:
            raw = archive.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON: raise ValueError('JSON expansion bound')
        records = json.loads(raw, object_pairs_hook=_pairs, parse_float=_float,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        if not isinstance(records, list) or not records: raise ValueError('Expected nonempty parent list')
        for record in records: _object(record, 'record')
        assays = [a for r in records for a in r['assays']]
        counts = {'parents': len(records), 'assays': len(assays),
                  'representative_assays': sum(a['is_representative'] for a in assays),
                  'measurements': sum(len(a['measurements']) for a in assays)}
        if expected_counts is not None and tuple(counts[k] for k in ('parents', 'assays', 'representative_assays')) != expected_counts:
            raise ValueError('Published counts mismatch')
    except (OSError, ValueError, TypeError, KeyError, EOFError, RecursionError, zlib.error):
        return outcome('ERROR', None, [source], LIMITATIONS, None, 'Activity archive integrity, bounds, schema or counts failed; no data extracted or substituted.')
    return outcome('SUCCESS', {'records': records, 'schema_version': '4.2', 'sha256': digest,
                              'counts': counts, 'source': source}, [source], LIMITATIONS, None)


def iter_assays(snapshot: ActivitySnapshot, substrate: str | None = 'PET') -> Iterator[AssayReference]:
    """Yield one AssayReference per assay, filtered by substrate (default PET).

    Purpose: lazy positional walk over the snapshot; each reference
        carries parent/assay indexes, protein identity, publication DOI,
        the assay payload, and a JSON-pointer source locator. None
        disables the substrate filter.
    Flow: Receives ActivitySnapshot from load_pazy_activity_snapshot(); yields AssayReference values for audit_pazy_pet_assays(), identify_cross_source_overlap() and describe_pazy_assay()."""
    for parent_index, record in enumerate(snapshot['records']):
        for assay_index, assay in enumerate(record['assays']):
            if substrate is None or assay['substrate'] == substrate:
                yield {'parent_index': parent_index, 'assay_index': assay_index, 'protein_id': record['protein_id'],
                       'protein_name': record['protein_name'], 'publication_doi': record['literature_doi'], 'assay': assay,
                       'source': {**snapshot['source'], 'locator': f"{snapshot['source']['locator']}#/{parent_index}/assays/{assay_index}"}}
