"""Data-only dialogue contracts; no scientific, agent, provider or UI imports."""

from typing import Literal, NotRequired, TypedDict

from .contracts import ExperimentalWindow
from .decision_types import DecisionContractDiagnostics, DecisionSession, ExperimentalDecision
from .runtime_config import RuntimeLimitValues, SessionUsage

ClarificationIntent = Literal["IDENTIFIER", "CONFIRM_CANDIDATE", "CONDITIONS"]
IdentityOutcome = Literal["IDENTIFIED_SEQUENCE", "CANDIDATE_MATCH", "UNRESOLVED"]


class ClarificationState(TypedDict):
    objective: Literal["CHOOSE_BETWEEN_ENZYMES"]
    requested_labels: list[str]
    confirmed_labels: list[str | None]
    rejected_labels: list[list[str]]
    temperatures_c: list[float]
    maximum_ph: float
    pending_index: int | None
    pending_candidate: str | None
    pending_question: str


class ClarificationTurn(TypedDict):
    state: ClarificationState
    message: str
    triage_outcome: IdentityOutcome | None
    sequence_validation: str | None


class DialogueContext(ClarificationState):
    decision: NotRequired[DecisionSession]
    model_attempts: int
    tool_attempts: int
    assay_ph: float | None
    pet_form: Literal["CRYSTALLINE_POWDER", "AMORPHOUS_FILM"] | None
    focus_label: str | None
    pending_intent: ClarificationIntent | None
    scientific_status: Literal["PARTIAL"]
    usage: SessionUsage


class DialogueInput(TypedDict):
    decision_support: NotRequired[bool]
    context: DialogueContext | None
    text: str
    input_kind: Literal["ANSWER", "SEQUENCE", "REFERENCE"]


class EquipmentBounds(TypedDict):
    temperatures_c: list[float]
    maximum_ph: float


class ClarificationSelection(TypedDict):
    action: Literal["ASK_CLARIFICATION"]
    target: str
    request: ClarificationIntent
    candidate: str | None
    question: str
    fact_ids: list[str]
    preserved_constraints: NotRequired[EquipmentBounds]


class ClarificationDecision(TypedDict):
    action: Literal["ASK_CLARIFICATION"]
    target: str
    request: ClarificationIntent
    candidate: str | None
    question: str
    presentation: Literal["MODEL_VALIDATED", "DETERMINISTIC_FALLBACK"]
    fallback_reason: str | None


class ClarificationDiagnostics(TypedDict):
    failure_code: str | None
    submitted_ids: int | None
    required_ids: int
    omitted_ids: int | None
    action: Literal["ASK_CLARIFICATION", "OTHER", "ABSENT"]


class IdentityFactView(TypedDict):
    id: str
    text: str
    evidence_category: str | None


class DialogueReport(TypedDict):
    decision_contract: NotRequired[DecisionContractDiagnostics]
    decision: NotRequired[ExperimentalDecision]
    context: DialogueContext
    outcome: Literal[
        "ASK_CLARIFICATION", "SCIENTIFIC_FINDINGS", "EXPERIMENTAL_DECISION", "INCOMPLETE"
    ]
    clarification: ClarificationDecision | None
    identity_facts: list[IdentityFactView]
    diagnostics: ClarificationDiagnostics


class LiveInvestigationInput(TypedDict):
    question: str
    completion_tokens: NotRequired[int]
    dialogue_input: NotRequired[DialogueInput]
    runtime_limits: NotRequired[RuntimeLimitValues]
    cohort_origin: NotRequired[str]
    cohort_filter: NotRequired[str]
    cohort_compare_filters: NotRequired[list[str]]
    cohort_compare_scopes: NotRequired[list[bool]]
    automatic_runtime: NotRequired[bool]
    cohort_pazy_only: NotRequired[bool]
    cohort_fingerprint: NotRequired[str]
    cohort_window: NotRequired[ExperimentalWindow]
    context_turns: NotRequired[list[str]]
