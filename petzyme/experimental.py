"""Source-bound metadata lookup and stable observation references; no file/network I/O."""
from __future__ import annotations

from .assessment_types import ExperimentalMetadata, MetadataRecord
from .contracts import Condition, Observation, Source
from .results import finite_number, sources_from_observations

PROTOCOL_FIELDS = ('incubation_duration', 'enzyme_loading', 'substrate_loading',
                   'buffer', 'detection_method', 'normalization')
LIMITATIONS = [
    'Missing metadata means not established from the supplied evidence, not that a procedure was absent.',
    'Metadata documentation is caller-supplied, source-bound evidence; it is not independently authenticated.',
    'Assessments are COMPUTED; original observations remain MEASURED and units are never converted.',
]


def cell_locator(observation: Observation) -> str:
    """Build the canonical cell locator 'dataset-locator#record=N&column=C'.

    Purpose: stable join key between observations and caller-supplied
        metadata records; the SHA-256 is paired with it at lookup time.
    Flow: Receives Observation from parse_nrel(); returns str for assess_evidence() and check_comparability() metadata lookup."""
    return f'{observation["source"]["locator"]}#record={observation["source_row"]}&column={observation["source_column"]}'


def source_key(observation: Observation) -> tuple[str, str]:
    """Return the (dataset-id, csv-sha256) identity of an observation's source.

    Purpose: detects when a comparison mixes datasets or file versions;
        DOI preferred, URL fallback, always paired with the file hash.
    Flow: Receives Observation from parse_nrel(); returns tuple[str, str] source identity for assess_evidence(), check_comparability() and investigate_conditions()."""
    source = observation['source']
    return source['doi'] or source['url'], observation['csv_sha256']


def unique_observations(observations: list[Observation]) -> list[Observation]:
    """Deduplicate observations to one entry per source cell.

    Purpose: repeated references to the same cell are collapsed so they
        can never masquerade as independent replicates. Two different
        payloads for one cell raise, since that indicates corruption.
    Raises:
        ValueError: on conflicting observations at one source cell.
    Flow: Receives list[Observation] from filter_observations(); returns list[Observation] of unique cells for assess_evidence(), check_comparability() and investigate_conditions()."""
    cells: dict[tuple, Observation] = {}
    for observation in observations:
        key = (*source_key(observation), observation['source_row'], observation['source_column'])
        if key in cells and cells[key] != observation:
            raise ValueError('Conflicting observations at one source cell')
        cells[key] = observation
    return list(cells.values())


def metadata_index(records: list[MetadataRecord] | None) -> dict[tuple[str, str], ExperimentalMetadata]:
    """Index caller metadata by (cell locator, CSV SHA-256) with strict validation.

    Purpose: binds each documentation claim to one exact cell of one exact
        file version. Records without complete documentation provenance are
        kept but emptied, so undocumented assertions establish nothing.
    Raises:
        ValueError: on duplicate cells, unknown fields, mistyped flags,
            invalid protocol entries, or bad replicate counts.
    Flow: Receives list[MetadataRecord] | None from assess_evidence() or check_comparability(); returns dict[tuple[str, str], ExperimentalMetadata] for details_for()."""
    index: dict[tuple[str, str], ExperimentalMetadata] = {}
    for record in records or []:
        key = record['cell_locator'], record['csv_sha256']
        if key in index:
            raise ValueError('Duplicate metadata records')
        details = record['details']
        if not isinstance(details, dict):
            raise ValueError('Invalid experimental metadata')
        for field, value in details.items():
            if field not in ExperimentalMetadata.__annotations__:
                raise ValueError('Unrecognized experimental metadata field')
            if field in ('units_verified', 'publication_mapping_verified'):
                if type(value) is not bool:
                    raise ValueError('Expected a documented verification flag')
            elif field == 'protocol':
                if (not isinstance(value, dict) or any(key not in PROTOCOL_FIELDS or not isinstance(text, str) or not text.strip()
                                                       for key, text in value.items())):
                    raise ValueError('Invalid protocol metadata')
            elif field != 'replicates' and (not isinstance(value, str) or not value.strip()):
                raise ValueError('Expected nonempty documented metadata text')
        if 'replicates' in details and (type(details['replicates']) is not int or details['replicates'] < 1):
            raise ValueError('Invalid documented replicate count')
        documented = bool(record['documentation']) and all(
            source['locator'] and source['url'] and source['attribution'] for source in record['documentation'])
        # Undocumented or differently located assertions do not establish facts.
        index[key] = details if documented else {}
    return index


def details_for(observation: Observation, index: dict[tuple[str, str], ExperimentalMetadata]) -> ExperimentalMetadata:
    """Look up the documented metadata for one observation's cell ({} when absent).

    Flow: Receives Observation and metadata_index() lookup dict; returns ExperimentalMetadata for assess_evidence() and check_comparability()."""
    return index.get((cell_locator(observation), observation['csv_sha256']), {})


def provenance(observations: list[Observation], records: list[MetadataRecord] | None) -> list[Source]:
    """Collect dataset sources plus documentation sources for the exact cells used.

    Purpose: provenance for an assessment covers both the observations and
        only the metadata records that actually attach to those cells.
    Flow: Receives list[Observation] and list[MetadataRecord] | None from assess_evidence() or investigate_conditions(); returns list[Source] for outcome()."""
    sources = sources_from_observations(observations)
    keys = {(cell_locator(observation), observation['csv_sha256']) for observation in observations}
    for record in records or []:
        if (record['cell_locator'], record['csv_sha256']) in keys:
            for source in record['documentation']:
                if source not in sources:
                    sources.append(source)
    return sources


def condition_matches(observation: Observation, condition: Condition) -> bool:
    """Check exact temperature/pH/form equality between an observation and a Condition.

    Flow: Receives Observation from parse_nrel() and external Condition; returns bool for assess_evidence() coverage checks."""
    return all(observation[field] == condition[field] for field in ('temperature_c', 'ph', 'pet_form'))


def valid_condition(condition: Condition) -> bool:
    """Validate a Condition: finite temperature/pH plus a known PET form.

    Flow: Receives external Condition; returns bool for assess_evidence() and investigate_conditions() input validation."""
    return (finite_number(condition['temperature_c']) and finite_number(condition['ph']) and
            condition['pet_form'] in ('CRYSTALLINE_POWDER', 'AMORPHOUS_FILM'))
