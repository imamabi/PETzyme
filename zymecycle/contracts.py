"""JSON-compatible contracts shared by independent scientific tools."""
from __future__ import annotations

from typing import Generic, Literal, NotRequired, TypeAlias, TypedDict, TypeVar

JSONValue: TypeAlias = str | int | float | bool | None | list['JSONValue'] | dict[str, 'JSONValue']
Status = Literal['SUCCESS', 'NO_DATA', 'PARTIAL', 'ERROR']
Evidence = Literal['MEASURED', 'COMPUTED', 'SIMULATED', 'HYPOTHETICAL']
PetForm = Literal['CRYSTALLINE_POWDER', 'AMORPHOUS_FILM']
T = TypeVar('T')

SourceStatus = Literal['FOUND', 'EMPTY', 'EXCLUDED', 'NOT_RELEVANT', 'UNAVAILABLE', 'PENDING']
DISCOVERY_SCOPE_LIMIT = 'Discovery is bounded to one documented assay group; global performance and construct identity are not established.'


class SourceReceipt(TypedDict):
    source: Literal['NREL', 'PAZy']
    status: SourceStatus
    measurements: int
    query_attempts: int
    reason: str


class Source(TypedDict):
    name: str
    url: str
    doi: str | None
    locator: str | None
    attribution: str
    retrieved_at_utc: str | None


class ReferenceCounts(TypedDict):
    total: int | None
    returned: int
    omitted: int | None
    unresolved_returned: int


class ToolResult(TypedDict, Generic[T]):
    status: Status
    result: T | None
    provenance: list[Source]
    limitations: list[str]
    evidence_category: Evidence | None
    message: str | None
    reference_counts: NotRequired[ReferenceCounts]


class NrelTable(TypedDict):
    columns: list[str]
    rows: list[list[str]]
    source: Source
    csv_sha256: str


class SequenceFingerprint(TypedDict):
    normalized_sequence: str | None
    sha256: str | None
    length: int | None


class UnitRepresentation(TypedDict):
    status: Literal['INFERRED_SUPPORTED', 'UNRESOLVED']
    source_unit: str
    display_unit: str
    standardized_value: float | None
    conversion_factor: int | None
    original_declaration: str
    evidence_category: Literal['COMPUTED']
    workbook_rows: str
    method: str
    rounding_sensitive: bool
    reason: str

class Observation(TypedDict):
    enzyme_identifier: str | None
    sequence_sha256: str | None
    sequence_reference: str
    source_row: int
    source_column: str
    source: Source
    csv_sha256: str
    temperature_c: float
    ph: float
    pet_form: PetForm
    original_text: str
    numerical_value: float
    unit_status: Literal['UNKNOWN/UNVERIFIED']
    is_zero: bool
    evidence_category: Literal['MEASURED']


class ParsedObservations(TypedDict):
    observations: list[Observation]
    missing_cells: int


class Condition(TypedDict):
    temperature_c: float
    ph: float
    pet_form: PetForm


class ExperimentalWindow(TypedDict):
    temperatures_c: list[float]
    ph_values: list[float]
    minimum_ph: float | None
    maximum_ph: float | None
    pet_forms: list[PetForm]


class ConditionSummary(TypedDict):
    condition: Condition
    observation_count: int
    recorded_zeros: int
    positive_recorded_responses: int
    leading_responses: list[Observation]


class NrelProtein(TypedDict):
    enzyme_identifier: str
    amino_acid_sequence: str | None
    source_row: int
    source: Source
    csv_sha256: str


class LiteratureRecord(TypedDict):
    id: int
    doi: str | None
    author: str | None
    title: str | None
    year: int | None


class LiteratureReference(TypedDict):
    reference_id: int | None
    resolution: Literal['EMBEDDED', 'UNRESOLVED']
    record: LiteratureRecord | None


class Organism(TypedDict):
    id: int
    scientific_name: str | None
    domain: str | None
    phylum: str | None


class Substrate(TypedDict):
    id: int
    name: str | None
    abbreviation: str | None
    oligomer_only: bool | None


class PazyProtein(TypedDict):
    id: int
    name: str | None
    amino_acid_sequence: str | None
    uniprot_accession: str | None
    organism: Organism | None
    substrates: list[Substrate] | None
    literature: list[LiteratureReference] | None
    source: Source


class IdentityComparison(TypedDict):
    identity: Literal['EXACT_SEQUENCE_MATCH', 'DIFFERENT_SEQUENCE', 'INSUFFICIENT_INFORMATION']
    nrel_identifier: str
    pazy_identifier: int
    nrel_sequence_sha256: str | None
    pazy_sequence_sha256: str | None
    nrel_sequence_length: int | None
    pazy_sequence_length: int | None
    normalization: str
