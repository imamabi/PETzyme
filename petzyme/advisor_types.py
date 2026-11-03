"""JSON contracts for proposals; no proposed value represents a measured result."""
from typing import Literal, TypedDict
from .contracts import Condition, Source

ActionCategory = Literal['EVIDENCE_VERIFICATION', 'PROPOSED_EXPERIMENT']
Resource = Literal['SOURCE_DOCUMENTATION', 'ASSAY_PROTOCOL', 'ENZYME_MATERIAL',
                   'PET_MATERIAL', 'TEMPERATURE_CONTROL', 'PH_CONTROL',
                   'RESPONSE_MEASUREMENT', 'LAB_ACCESS']


class ResearcherConstraints(TypedDict, total=False):
    available_resources: list[Resource]
    unavailable_resources: list[Resource]
    budget: Literal['AVAILABLE', 'UNAVAILABLE', 'UNKNOWN']
    costs: Literal['DOCUMENTED', 'UNKNOWN']
    turnaround: Literal['ACCEPTABLE', 'UNACCEPTABLE', 'UNKNOWN']
    allow_new_measurements: bool


class EvidenceBasis(TypedDict):
    category: Literal['MEASURED', 'COMPUTED']
    statement: str
    sources: list[Source]
    csv_sha256: str | None


class ExperimentCandidate(TypedDict):
    action_id: str
    code: str
    category: ActionCategory
    evidence_category: Literal['HYPOTHETICAL']
    objective: str
    evidence_basis: list[EvidenceBasis]
    proposed_action: str
    required_measurements: list[str]
    target_conditions: list[Condition]
    controlled_variables: list[str]
    varied_variables: list[str]
    expected_information: str
    prerequisites: list[str]
    required_resources: list[Resource]
    information_value: Literal['RESOLVES_BLOCKER', 'DIRECT_QUESTION']
    limitations: list[str]


class FeasibilityAssessment(TypedDict):
    action_id: str
    status: Literal['UNKNOWN', 'BLOCKED', 'CONFLICTING_CONSTRAINTS']
    blockers: list[str]
    unknown_requirements: list[str]
    basis: str


class Recommendation(ExperimentCandidate):
    feasibility: FeasibilityAssessment
    priority: int
    ranking_rationale: str


class ExperimentPlan(TypedDict):
    recommendations: list[Recommendation]
    candidate_count: int
    omitted_count: int
    omitted_categories: list[ActionCategory]
    basis_kind: Literal['CONDITION_INVESTIGATION', 'SEQUENCE_REPRESENTATION', 'EXPERIMENTAL_COVERAGE']
    scientific_status: Literal['PROPOSED_ONLY']
    limitations: list[str]
