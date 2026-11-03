"""JSON contracts for deterministic evidence and comparison instruments."""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from .contracts import Condition, Observation, Source


class ProtocolMetadata(TypedDict, total=False):
    incubation_duration: str
    enzyme_loading: str
    substrate_loading: str
    buffer: str
    detection_method: str
    normalization: str


class ExperimentalMetadata(TypedDict, total=False):
    experimental_group: str
    measurement_definition: str
    measurement_unit: str
    units_verified: bool
    protocol_id: str
    protocol: ProtocolMetadata
    detection_limit: str
    replicates: int
    uncertainty: str
    publication_doi: str
    publication_mapping_verified: bool


class MetadataRecord(TypedDict):
    cell_locator: str
    csv_sha256: str
    documentation: list[Source]
    details: ExperimentalMetadata


class EvidenceIssue(TypedDict):
    code: str
    severity: Literal['HIGH', 'MEDIUM', 'INFO']
    explanation: str
    affected_cells: list[str]
    provenance: list[Source]


class CoverageGap(TypedDict):
    enzyme_identifier: str | None
    condition: Condition
    source_row: int
    source: Source
    csv_sha256: str


class EvidenceAssessment(TypedDict):
    observation_count: int
    issues: list[EvidenceIssue]
    missing_conditions: list[CoverageGap]
    coverage_scope_explicit: bool


ComparabilityCategory = Literal['COMPARABLE_WITHIN_DOCUMENTED_GROUP', 'LIMITED_COMPARISON',
                                 'NOT_COMPARABLE', 'INSUFFICIENT_INFORMATION']
VaryingVariable = Literal['temperature_c', 'ph']


class NumericalComparison(TypedDict):
    original_a: str
    original_b: str
    difference_b_minus_a: str
    ratio_b_over_a: str | None
    percent_change_from_a: str | None
    unit: str
    evidence_category: Literal['COMPUTED']
    metadata_basis: Literal['CALLER_SUPPLIED_SOURCE_BOUND_NOT_INDEPENDENTLY_AUTHENTICATED']


class ComparabilityAssessment(TypedDict):
    category: ComparabilityCategory
    observation_count: int
    differing_fields: list[str]
    blocking_differences: list[str]
    missing_fields: list[str]
    varying_variable: VaryingVariable | None
    descriptive_within_column_supported: bool
    quantitative_comparison_supported: bool
    issues: list[EvidenceIssue]
    numerical_comparison: NotRequired[NumericalComparison]


class ConditionMeasurements(TypedDict):
    condition: Condition
    observations: list[Observation]
    status: Literal['OBSERVED', 'MISSING']


class ExperimentalRecordIdentity(TypedDict):
    enzyme_identifier: str | None
    source_row: int
    csv_sha256: str
    source: Source


class InvestigationIdentity(TypedDict, total=False):
    record_identity: ExperimentalRecordIdentity | None


class ConditionInvestigation(InvestigationIdentity):
    enzyme_identifier: str | None
    conditions: list[ConditionMeasurements]
    changed_variables: list[str]
    comparability: ComparabilityAssessment
    evidence: EvidenceAssessment
    direct_comparison_supported: bool
    causal_attribution_supported: bool
    uncertainties: list[str]
    unsupported: list[str]
    source_provenance: list[Source]


class EvidenceGap(TypedDict):
    priority: int
    code: str
    kind: Literal['VERIFICATION', 'MEASUREMENT_REQUIREMENT']
    requirement: str
    target_conditions: list[Condition]
