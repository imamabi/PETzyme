"""Data-only contracts for hypothetical experimental choices and safe continuation."""

from typing import Literal, TypedDict

from .advisor_types import ExperimentCandidate, FeasibilityAssessment
from .assessment_types import ComparabilityAssessment, EvidenceAssessment
from .contracts import Source

Design = Literal[
    "MATCHED_ENZYME_COMPARISON", "BALANCED_ENZYME_TEMPERATURE", "MEASUREMENT_QUALIFICATION"
]
Criterion = Literal[
    "EQUIPMENT_FEASIBILITY",
    "TEMPERATURE_COVERAGE",
    "FOCUSED_DESIGN",
    "DIRECT_ENZYME_DECISION",
    "MEASUREMENT_READINESS",
]
Control = Literal[
    "SUBSTRATE_BLANK",
    "REFERENCE_CONTROL",
    "INDEPENDENT_REPLICATION",
    "UNCERTAINTY",
    "DETECTION_LIMIT",
    "MEASUREMENT_CALIBRATION",
]


class ExperimentOption(TypedDict):
    option_id: str
    design: Design
    enzymes: list[str]
    temperatures_c: list[float]
    ph: float
    pet_form: Literal["CRYSTALLINE_POWDER", "AMORPHOUS_FILM"]
    controlled_variables: list[str]
    varied_variables: list[str]
    controls: list[Control]
    evidence_ids: list[str]


class Preference(TypedDict):
    preferred: str
    over: str
    criterion: Criterion
    evidence_ids: list[str]


class DecisionSelection(TypedDict):
    action: Literal["EXPERIMENTAL_DECISION"]
    alternatives: list[ExperimentOption]
    ranking: list[str]
    rationale: list[Preference]
    fact_ids: list[str]


class NrelCellReference(TypedDict):
    csv_sha256: str
    source_row: int
    source_column: str


class DecisionSession(TypedDict):
    original_temperatures_c: list[float]
    nrel_references: list[NrelCellReference]
    omitted_reference_count: int
    previous_alternatives: list[ExperimentOption]
    previous_ranking: list[str]
    previous_rationale: list[Preference]
    pazy_context_seen: bool


class PublicSource(TypedDict):
    name: str
    url: str
    doi: str | None


class PublicFact(TypedDict):
    id: str
    kind: str
    text: str
    evidence_category: str | None
    sources: list[PublicSource]


class AssessedOption(TypedDict):
    proposal: ExperimentOption
    candidate: ExperimentCandidate
    equipment_status: Literal["WITHIN_DECLARED_BOUNDS", "BLOCKED"]
    equipment_blockers: list[str]
    resource_feasibility: FeasibilityAssessment
    identity_status: dict[str, str]
    metadata_status: dict[str, str]
    condition_coverage: dict[str, list[str]]
    proposed_ph_origin: str
    hypothesis: str
    outcome_implications: list[str]
    limitations: list[str]


class ExperimentalDecision(TypedDict):
    status: Literal["PARTIAL"]
    evidence_category: Literal["HYPOTHETICAL"]
    alternatives: list[AssessedOption]
    selected_id: str
    feasible_option_ids: list[str]
    excluded_ids: list[str]
    ranking: list[str]
    rationale: list[Preference]
    rationale_explanations: list[str]
    evidence_assessment: EvidenceAssessment
    comparability: ComparabilityAssessment
    previous_option_status: dict[str, str]
    retained_reference_count: int
    source_provenance: list[Source]
    evidence_facts: list[PublicFact]
    limitations: list[str]


SchemaIssueCode = Literal[
    "MISSING_FIELD",
    "UNEXPECTED_FIELD",
    "WRONG_TYPE",
    "UNSUPPORTED_ENUM",
    "INVALID_IDENTIFIER",
    "INVALID_COUNT",
    "NON_FINITE_NUMBER",
]


class DecisionSchemaIssue(TypedDict):
    section: Literal["decision", "option", "preference"]
    option_index: int | None
    preference_index: int | None
    field: str
    code: SchemaIssueCode
    expected: str
    received_type: str | None


class DecisionSchemaCheck(TypedDict):
    issues: list[DecisionSchemaIssue]
    omitted_issue_count: int
    recoverable: bool


class DecisionContractDiagnostics(TypedDict):
    original_schema_valid: bool | None
    original_accepted: bool
    initial_issues: list[DecisionSchemaIssue]
    initial_omitted_issue_count: int
    correction_attempts: int
    correction_status: Literal[
        "NOT_REQUESTED", "NOT_RECOVERABLE", "BUDGET_UNAVAILABLE", "ACCEPTED", "REJECTED"
    ]
    corrected_accepted: bool
    final_schema_valid: bool | None
    final_issues: list[DecisionSchemaIssue]
    final_omitted_issue_count: int
