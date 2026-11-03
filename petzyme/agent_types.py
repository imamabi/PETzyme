"""Provider-neutral, in-memory research agent contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, NotRequired, Protocol, TypedDict

from .advisor_types import ResearcherConstraints
from .contracts import ExperimentalWindow, JSONValue, Observation, Source, SourceReceipt, ToolResult
from .dialogue_types import DialogueContext, DialogueReport
from .runtime_config import DEFAULT_TURN_TOOLS, RuntimeLimitValues
from .scientific_scope import RequestScope, ScopeAssessment

CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


class ReasoningError(RuntimeError):
    """Provider-neutral sanitized reasoning/backend failure."""

    code = "BACKEND_FAILURE"


class ResearchRequest(TypedDict):
    window: NotRequired[ExperimentalWindow]
    question: str
    sources: list[str]
    objectives: list[str]
    enzyme_identifiers: list[str]
    temperatures: list[float]
    ph_values: list[float]
    pet_forms: list[str]
    scientific_scope: RequestScope
    context_scope: NotRequired[RequestScope]


class Fact(TypedDict):
    id: str
    call_id: str
    kind: Literal["EVIDENCE", "PROPOSAL", "LIMITATION", "UNANSWERED"]
    text: str
    evidence_category: str | None
    provenance: list[Source]
    ephemeral: bool


class ToolEvent(TypedDict):
    error_code: NotRequired[str | None]
    actor: NotRequired[Literal['MODEL', 'APPLICATION_COVERAGE']]
    call_id: str
    name: str
    arguments: dict[str, JSONValue]
    status: str
    refs: dict[str, str]
    preview: dict[str, JSONValue]
    fact_ids: list[str]
    ephemeral: bool


class ToolUpdate(TypedDict):
    error_code: NotRequired[str | None]
    actor: NotRequired[Literal['MODEL', 'APPLICATION_COVERAGE']]
    call_id: str
    name: str
    status: str
    summary: str
    model_requests: int
    tool_attempts: int
    data: dict[str, JSONValue]
    limitations: list[str]
    provenance: list[Source]


class Validation(TypedDict):
    grounded: bool
    complete: bool
    issues: list[str]
    selected_fact_ids: list[str]
    selection_counts: NotRequired[dict[str, int]]
    scientific_scope: NotRequired[ScopeAssessment]


class RoutingTrace(TypedDict):
    sources: list[str]
    objectives: list[str]
    exposed_tools: list[str]


class PlanDiagnostics(TypedDict):
    status: Literal['DETERMINISTIC', 'MUSE_VALIDATED', 'REJECTED']
    model_attempts: int
    scientific_claims_allowed: Literal[False]


class ResearchReport(TypedDict):
    interpretation: NotRequired[PlanDiagnostics]
    source_coverage: NotRequired[list[SourceReceipt]]
    routing: NotRequired[RoutingTrace]
    question: str
    status: Literal["COMPLETE", "PARTIAL"]
    provider_mode: str
    model_requests: int
    tool_attempts: int
    validation: Validation
    facts: list[Fact]
    trace: list[ToolEvent]
    limitations: list[str]
    persistence: Literal["NREL_ONLY", "FORBIDDEN_PAZY_DERIVED"]
    synthesis: NotRequired[dict[str, JSONValue]]
    communication: NotRequired[dict[str, JSONValue]]
    dialogue: NotRequired[DialogueReport]
    limits: NotRequired[RuntimeLimitValues]


class ReasoningBackend(Protocol):
    mode: str

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """Flow: Receives messages and tool list from run_investigation(); returns assistant dict for its next step or raises ReasoningError."""
        ...


@dataclass(repr=False)
class AgentState:
    """One investigation only; never serialize or log this private artifact store."""

    csv_path: str
    artifacts: dict[str, tuple[str, object]] = field(default_factory=dict)
    results: dict[str, ToolResult] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    events: list[ToolEvent] = field(default_factory=list)
    signatures: set[str] = field(default_factory=set)
    call_ids: set[str] = field(default_factory=set)
    sequences: set[str] = field(default_factory=set)
    live_protein_ids: set[int] = field(default_factory=set)
    decision_observations: list[Observation] = field(default_factory=list)
    tool_attempts: int = 0
    researcher_constraints: ResearcherConstraints | None = None
    request_window: ExperimentalWindow | None = None
    dialogue_context: DialogueContext | None = None
    tool_limit: int | None = DEFAULT_TURN_TOOLS
