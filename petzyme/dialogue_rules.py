"""Agent-side dialogue validation over typed contracts and executed scientific tools."""

from __future__ import annotations

import math
from typing import Any, TypeGuard, cast

from .agent_request import model_tool_description, select_tool_catalog
from .agent_types import AgentState, ResearchRequest, Validation
from .clarification_intent import MAX_QUESTION_CHARS, question_presentation
from .decision_context import valid_decision_session
from .dialogue_types import (
    ClarificationDecision,
    ClarificationDiagnostics,
    ClarificationSelection,
    DialogueContext,
)
from .privacy import identifier_safe as _label
from .privacy import outbound_safe
from .protein_triage import IdentityTriage, triage_protein
from .tools import describe_tools


def validate_dialogue_context(context: object) -> TypeGuard[DialogueContext]:
    """Type-guard a retained dialogue context: exact keys, safe labels, bounded numbers.

    Purpose: rejects stale, partial, or tampered session contexts
        (including unbounded usage or unsafe label strings) before they
        can steer a new turn; callers must reset explicitly instead.
    Flow: Receives external worker/session context object; returns bool for prepare_dialogue_turn() before context is forwarded."""
    if (
        not isinstance(context, dict)
        or not DialogueContext.__required_keys__
        <= context.keys()
        <= DialogueContext.__annotations__.keys()
    ):
        return False
    if "decision" in context:
        if not valid_decision_session(context["decision"]):
            return False
    if context["objective"] != "CHOOSE_BETWEEN_ENZYMES":
        return False
    if any(type(context[k]) is not int or context[k] < 0
           for k in ("model_attempts", "tool_attempts")):
        return False
    if context["scientific_status"] != "PARTIAL" or context["pending_intent"] not in (
        None,
        "IDENTIFIER",
        "CONFIRM_CANDIDATE",
        "CONDITIONS",
    ):
        return False
    usage = context["usage"]
    usage_ints = (
        "actual_http_requests",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "unknown_usage_requests",
        "unknown_reasoning_requests",
    )
    if not isinstance(usage, dict) or set(usage) != set(usage_ints) | {
        "reserved_cost_usd",
        "elapsed_seconds",
    }:
        return False
    if any(type(usage[k]) is not int or usage[k] < 0 for k in usage_ints):
        return False
    if any(
        type(usage[k]) not in (int, float) or not math.isfinite(usage[k]) or usage[k] < 0
        for k in ("reserved_cost_usd", "elapsed_seconds")
    ):
        return False
    labels, confirmed, rejected = (
        context[k] for k in ("requested_labels", "confirmed_labels", "rejected_labels")
    )
    if not all(isinstance(x, list) and len(x) == 2 for x in (labels, confirmed, rejected)):
        return False
    if not all(_label(x) for x in labels) or labels[0] == labels[1]:
        return False
    if not all(x is None or _label(x) for x in confirmed):
        return False
    if not all(
        isinstance(xs, list) and len(xs) <= 5 and all(_label(x) for x in xs) for xs in rejected
    ):
        return False
    ts, ph = context["temperatures_c"], context["maximum_ph"]
    if (
        not isinstance(ts, list)
        or not 1 <= len(ts) <= 8
        or any(type(t) not in (int, float) or not math.isfinite(t) or t < -273.15 for t in ts)
        or type(ph) not in (int, float)
        or not math.isfinite(ph)
        or not 0 <= ph <= 14
    ):
        return False
    if "decision" in context and not set(ts) <= set(context["decision"]["original_temperatures_c"]):
        return False
    exact = context["assay_ph"]
    if exact is not None and (
        type(exact) not in (int, float) or not math.isfinite(exact) or not 0 <= exact <= ph
    ):
        return False
    if context["pet_form"] not in (None, "CRYSTALLINE_POWDER", "AMORPHOUS_FILM"):
        return False
    if context["focus_label"] is not None and context["focus_label"] not in confirmed:
        return False
    if context["pending_index"] is not None and type(context["pending_index"]) is not int:
        return False
    if context["pending_index"] not in (None, 0, 1):
        return False
    if context["pending_candidate"] is not None and not _label(context["pending_candidate"]):
        return False
    if (
        not isinstance(context["pending_question"], str)
        or len(context["pending_question"]) > MAX_QUESTION_CHARS
        or not outbound_safe(context["pending_question"])
    ):
        return False
    # Only labels proven against the current local NREL source can be carried as established/rejected.
    for i in (0, 1):
        if confirmed[i] in rejected[i] and confirmed[i] is not None:
            return False
        for name in [x for x in [confirmed[i], *rejected[i]] if x is not None]:
            result = triage_protein(name=name)["result"]
            if result is None:
                return False
            cards = result["candidates"]
            if not any(
                c["source_name"] == "NREL" and c["name"] == name and "EXACT_NAME" in c["basis"]
                for c in cards
            ):
                return False
    return True


def dialogue_catalog(request: ResearchRequest) -> list[dict[str, Any]]:
    """Build the three-tool dialogue catalog (triage, PET evidence, advice).

    Purpose: dialogue turns may use only triage_protein,
        investigate_pet_evidence, and advise_next_experiment, each with
        dialogue-specific argument guards (opaque sequence ref, no
        unestablished labels, no invented conditions).
    Flow: Receives ResearchRequest from run_investigation(); returns privacy-restricted tool definitions for that loop's Muse decisions."""
    tools: list[dict[str, Any]] = []
    for name in ("investigate_pet_evidence", "advise_next_experiment"):
        r: ResearchRequest = {
            **request,
            "sources": ["NREL", "PAZy"],
            "objectives": ["multi_study"]
            + (["experiment_advice"] if name == "advise_next_experiment" else []),
        }
        tools.extend(t for t in select_tool_catalog(r) if t['function']['name'] == name)
    descriptor = next(
        d for d in (describe_tools()["result"] or []) if d["name"] == "triage_protein"
    )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "triage_protein",
                "description": model_tool_description(descriptor["description"]),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "maxLength": 120},
                        "sequence_input_ref": {"type": "string", "enum": ["turn:sequence"]},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": False,
            },
        }
    )
    return tools


def sync_identity_context(state: AgentState, call_id: str, arguments: dict[str, Any]) -> None:
    """Sync a real triage outcome into the dialogue's minimal label context.

    Purpose: records established/unresolved NREL labels and rejections
        from the executed result only. Native records, sequences, and
        PAZy selections never enter retained context.
    Flow: Receives real triage ToolResult in AgentState from execute_tool_call(); returns None after updating minimal NREL labels in active dialogue context."""
    context = state.dialogue_context
    if context is None:
        return
    value = cast(IdentityTriage | None, state.results[call_id]["result"])
    if value is None:
        return
    cards = value["candidates"]
    for i, label in enumerate(context["requested_labels"]):
        if context["confirmed_labels"][i] is not None:
            continue
        exact = {
            c["name"]
            for c in cards
            if c["source_name"] == "NREL"
            and (
                (
                    "EXACT_NAME" in c["basis"]
                    and c["name"] == label
                    and arguments.get("name") == label
                )
                or (
                    "EXACT_SEQUENCE" in c["basis"]
                    and value["identity_status"] == "IDENTIFIED_SEQUENCE"
                    and context["pending_index"] == i
                )
            )
        }
        if len(exact) == 1 and not value["omitted_candidates"]:
            name = next(iter(exact))
            if name not in context["rejected_labels"][i]:
                context["confirmed_labels"][i] = name
    context["pending_index"] = next(
        (i for i, x in enumerate(context["confirmed_labels"]) if x is None), None
    )


def guard_dialogue_tool(name: str, resolved: dict[str, Any], state: AgentState) -> None:
    """Block out-of-scope dialogue tool arguments before execution.

    Purpose: rejects unestablished/rejected labels, invented pH/PET
        conditions, and unavailable temperatures so the advisor and
        evidence tools can only run on validated dialogue scope.
    Raises:
        ValueError: on any out-of-scope argument.
    Flow: Receives resolved tool arguments from execute_tool_call(); returns None or raises ValueError before any out-of-scope scientific execution."""
    c = state.dialogue_context
    if c is None or name == "triage_protein":
        return
    confirmed = {x for x in c["confirmed_labels"] if x is not None}
    rejected = {x for group in c["rejected_labels"] for x in group}
    if "decision" in c:
        # Exploratory source filters are not requested assay setpoints or proposed outcomes.
        if name == "investigate_pet_evidence":
            q = resolved.get("query") or {}
            if (
                set(q) - {"protein_name", "temperature_c", "ph", "form"}
                or q.get("protein_name") not in confirmed
                or q["protein_name"] in rejected
            ):
                raise ValueError("Experimental decision evidence requires a confirmed source label")
            return
        if name == "advise_next_experiment":
            if (
                resolved.get("enzyme_identifier") not in confirmed
                or resolved["enzyme_identifier"] in rejected
            ):
                raise ValueError("Conditional per-enzyme advice requires a confirmed label")
            return
    if name == "investigate_pet_evidence":
        q = resolved.get("query") or {}
        if (
            set(q) - {"protein_name", "temperature_c", "ph", "form"}
            or q.get("protein_name") not in confirmed
            or q["protein_name"] in rejected
            or c["focus_label"] is not None
            and q["protein_name"] != c["focus_label"]
            or "temperature_c" in q
            and q["temperature_c"] not in c["temperatures_c"]
            or "ph" in q
            and (c["assay_ph"] is None or q["ph"] != c["assay_ph"])
            or "form" in q
            and q["form"]
            != {"CRYSTALLINE_POWDER": "powder", "AMORPHOUS_FILM": "film"}.get(c["pet_form"] or "")
        ):
            raise ValueError("Unconfirmed identity or invented exact conditions")
    elif name == "advise_next_experiment":
        if c["assay_ph"] is None or c["pet_form"] is None:
            raise ValueError("Exact requested assay conditions are missing")
        label = resolved.get("enzyme_identifier")
        if (
            label not in confirmed
            or label in rejected
            or c["focus_label"] is not None
            and label != c["focus_label"]
        ):
            raise ValueError("Unconfirmed or wrong focused identity")
        conditions = [resolved.get("condition_a") or {}, resolved.get("condition_b") or {}]
        if any(
            x.get("temperature_c") not in c["temperatures_c"]
            or x.get("ph") != c["assay_ph"]
            or x.get("pet_form") != c["pet_form"]
            for x in conditions
        ):
            raise ValueError("Invented or unavailable assay conditions")
    else:
        raise ValueError("Unsupported dialogue scientific capability")


def required_dialogue_fact_ids(state: AgentState) -> list[str]:
    """List every executed EVIDENCE/PROPOSAL fact ID as the dialogue checklist.

    Purpose: stricter than the standard checklist — all executed
        source/proposal IDs are mandatory, so warning-only selections
        cannot masquerade as complete evidence.
    Flow: Receives executed AgentState from run_investigation(); returns list[str] of every source evidence/proposal ID for its checklist and final validation."""
    return [f["id"] for f in state.facts if f["kind"] in ("EVIDENCE", "PROPOSAL")]


POLICY = """Dialogue is the same scientific agent, using the supplied finite turn/session limits.
Decide when identity tools, PET evidence/advice, clarification or an honest withheld conclusion are useful.
Use triage_protein(name=...) for source-backed identity candidates. Supplied local sequences MUST be
triaged through sequence_input_ref="turn:sequence" before you describe their validation or ask a follow-up.
Never request or forward sequence bytes/hashes, embeddings, datasets or native PAZy records.
Source-listed labels and exact reference matches do not establish experimental construct identity or PET activity.
Preserve the original two-enzyme objective, rejected labels and equipment constraints. An upper pH bound
is not an assay setpoint. Only explicitly supplied exact conditions may be used for bounded per-enzyme advice;
the existing advisor does not perform a validated two-enzyme ranking.
ASK_CLARIFICATION is a separate structured outcome, not a forced experimental final answer. Return JSON:
{"action":"ASK_CLARIFICATION","target":"an original requested label","request":"IDENTIFIER",
 "candidate":null,"question":"your natural focused information request","fact_ids":["all current required IDs"]}.
request may be IDENTIFIER, CONFIRM_CANDIDATE or CONDITIONS. A plain request for identifying information
makes no claim that a source lookup failed; it need not manufacture a tool result or a candidate.
CONFIRM_CANDIDATE requires a real current source-backed NREL candidate for that target, never a guessed alias.
For a requested enzyme comparison, prioritize unresolved identities before asking for assay conditions.
When any original requested label is unresolved and there is no explicit single-enzyme focus, select
IDENTIFIER or a source-backed CONFIRM_CANDIDATE for an unresolved target; do not jump to CONDITIONS.
CONDITIONS requests exact pH and PET form for an established reference label after that identity step.
You choose the unresolved target and tool calls; no biological identity can be inferred from spelling.
You formulate natural wording. Put only the focused information request in question, without prefacing
it with scientific findings; those are displayed separately from canonical facts. It is fine to mention
both original enzymes as comparison context without asserting that they are aliases or equivalents.
Do not re-propose rejected identities. Avoid repeating measurements or sequence-validation facts in
question; local triage displays those separately. No exact sentence wording is prescribed.
You formulate natural wording. Ask for the missing information; do not embed unverified identity, numerical,
activity, causal or sequence-to-function assertions. Scientific facts are rendered separately from canonical
fact references. Harmless equivalent wording is permitted. Unverifiable prose will be withheld in favor of
an explicitly labelled grounded fallback, without pretending that fallback is your scientific reasoning.
Optional preserved_constraints must exactly match the external equipment bounds, without inventing an assay pH.
Select every identity_required_fact_ids/final_selection required ID already returned this turn, including
source evidence and HYPOTHETICAL proposals. Never auto-infer scientific records. LIMITATION/UNANSWERED
facts are retained automatically. For scientific findings keep the existing fact_ids/explanation contract;
asking for more information remains scientifically PARTIAL, not proof of enzyme superiority.
"""


class ClarificationError(ValueError):
    """Safe category only; never a provider draft, identity payload or exception echo."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def parse_clarification_selection(value: object) -> ClarificationSelection:
    """Structurally validate a model's ASK_CLARIFICATION JSON object.

    Purpose: accepts exactly the action/target/request/candidate/
        question/fact_ids shape (plus optional preserved_constraints);
        anything else raises a safe ClarificationError carrying no
        provider content.
    Flow: Receives untrusted model JSON object from validate_clarification(); returns ClarificationSelection after structural validation or raises a safe ClarificationError."""
    keys = {"action", "target", "request", "candidate", "question", "fact_ids"}
    if not isinstance(value, dict) or not keys <= set(value) <= keys | {"preserved_constraints"}:
        raise ClarificationError("INVALID_CLARIFICATION_SCHEMA")
    if value["action"] != "ASK_CLARIFICATION" or value["request"] not in (
        "IDENTIFIER",
        "CONFIRM_CANDIDATE",
        "CONDITIONS",
    ):
        raise ClarificationError("INVALID_CLARIFICATION_INTENT")
    if not isinstance(value["target"], str) or not isinstance(value["question"], str):
        raise ClarificationError("INVALID_CLARIFICATION_SCHEMA")
    if value["candidate"] is not None and not isinstance(value["candidate"], str):
        raise ClarificationError("INVALID_CLARIFICATION_SCHEMA")
    ids = value["fact_ids"]
    if not isinstance(ids, list) or len(ids) > 100 or not all(isinstance(i, str) for i in ids):
        raise ClarificationError("INVALID_FACT_REFERENCES")
    return cast(ClarificationSelection, value)


def clarification_diagnostics(
    value: object, state: AgentState, failure_code: str | None = None
) -> ClarificationDiagnostics:
    """Flow: Receives transient model object and actual AgentState from run_investigation(); returns redacted ClarificationDiagnostics counts/categories, never prose or submitted IDs."""
    required = set(required_dialogue_fact_ids(state))
    raw = value if isinstance(value, dict) else {}
    ids = raw.get("fact_ids")
    valid_ids = isinstance(ids, list) and all(isinstance(i, str) for i in ids)
    return {
        "failure_code": failure_code,
        "action": "ASK_CLARIFICATION"
        if raw.get("action") == "ASK_CLARIFICATION"
        else "OTHER"
        if value is not None
        else "ABSENT",
        "submitted_ids": len(ids) if isinstance(ids, list) else None,
        "required_ids": len(required),
        "omitted_ids": len(required - set(cast(list[str], ids))) if valid_ids else None,
    }


def validate_clarification(
    value: object, state: AgentState
) -> tuple[Validation, ClarificationDecision]:
    """Validate a clarification decision: intent, facts, constraints, wording.

    Purpose: checks the selection's target/candidate against current
        source backing, requires all executed source/proposal IDs,
        verifies constraint preservation, and screens the question
        wording — replacing unverified prose with an explicitly
        labelled deterministic fallback. Yields PARTIAL
        VALIDATED_CLARIFICATION, never a completed answer.
    Flow: Receives model JSON object and real AgentState from run_investigation(); returns Validation and ClarificationDecision, keeping unverifiable prose out of the result."""
    selection = parse_clarification_selection(value)
    context = state.dialogue_context
    if context is None:
        raise ClarificationError("NO_DIALOGUE_CONTEXT")
    target, kind, candidate = selection["target"], selection["request"], selection["candidate"]
    if target not in context["requested_labels"]:
        raise ClarificationError("UNKNOWN_REQUESTED_TARGET")
    index = context["requested_labels"].index(target)
    facts = {fact["id"]: fact for fact in state.facts}
    ids = selection["fact_ids"]
    if not set(ids) <= facts.keys():
        raise ClarificationError("UNKNOWN_FACT_REFERENCES")
    required = required_dialogue_fact_ids(state)
    if not set(required) <= set(ids):
        raise ClarificationError("OMITTED_SOURCE_FACTS")
    if "preserved_constraints" in selection:
        bounds = selection["preserved_constraints"]
        if (
            not isinstance(bounds, dict)
            or set(bounds) != {"temperatures_c", "maximum_ph"}
            or not isinstance(bounds["temperatures_c"], list)
            or not all(type(t) in (int, float) for t in bounds["temperatures_c"])
            or set(bounds["temperatures_c"]) != set(context["temperatures_c"])
            or type(bounds["maximum_ph"]) not in (int, float)
            or bounds["maximum_ph"] != context["maximum_ph"]
        ):
            raise ClarificationError("CONSTRAINT_MISMATCH")
    events = [event for event in state.events if event["name"] == "triage_protein"]
    sequence_events = [
        e
        for e in events
        if e["arguments"].get("sequence_input_ref") == "turn:sequence"
        and state.results[e["call_id"]]["result"] is not None
    ]
    if "turn:sequence" in state.artifacts and not sequence_events:
        raise ClarificationError("LOCAL_SEQUENCE_NOT_TRIAGED")
    relevant = [
        event
        for event in events
        if event["arguments"].get("name") == target
        or context["confirmed_labels"][index] is not None
        and event["arguments"].get("name") == context["confirmed_labels"][index]
        or event in sequence_events
    ]
    if kind == "CONFIRM_CANDIDATE":
        if candidate is None or candidate in context["rejected_labels"][index]:
            raise ClarificationError("REJECTED_OR_MISSING_CANDIDATE")
        cards = [
            card
            for event in relevant
            for card in cast(
                IdentityTriage, state.results[event["call_id"]]["result"] or {"candidates": []}
            )["candidates"]
        ]
        if not any(card["source_name"] == "NREL" and card["name"] == candidate for card in cards):
            raise ClarificationError("CANDIDATE_NOT_SOURCE_BACKED")
    elif candidate is not None:
        raise ClarificationError("UNEXPECTED_CANDIDATE")
    elif kind == "CONDITIONS" and context["confirmed_labels"][index] is None:
        raise ClarificationError("CONDITIONS_REQUIRE_ESTABLISHED_LABEL")
    if context["focus_label"] is None and any(x is None for x in context["confirmed_labels"]):
        if kind == "CONDITIONS" or context["confirmed_labels"][index] is not None:
            raise ClarificationError("UNRESOLVED_IDENTITY_REQUIRES_ATTENTION")
    # Plain identifying questions do not assert a failed lookup or a biological identity;
    # requiring a fresh target lookup here conflated an information request with evidence.
    triages = [
        cast(IdentityTriage, state.results[e["call_id"]]["result"])
        for e in events
        if state.results[e["call_id"]]["result"] is not None
    ]
    numbers = [float(t["sequence_length"]) for t in triages if t["sequence_length"] is not None]
    symbols = [symbol for t in triages for symbol in t["flagged_symbols"] if symbol in "BJXZUO"]
    decision = question_presentation(
        selection, context, verified_numbers=numbers, verified_symbols=symbols
    )
    context["pending_index"] = index
    context["pending_candidate"] = candidate
    context["pending_question"] = decision["question"]
    context["pending_intent"] = kind
    context["scientific_status"] = "PARTIAL"
    warnings = [fact["id"] for fact in state.facts if fact["kind"] in ("LIMITATION", "UNANSWERED")]
    issues = [
        "Scientific clarification remains pending; no enzyme superiority or experimental conclusion is established."
    ]
    if decision["presentation"] == "DETERMINISTIC_FALLBACK":
        issues.append(
            "Unverified model wording was withheld; the displayed question is a deterministic grounded fallback."
        )
    if any(e["status"] == "ERROR" for e in state.events):
        issues.append(
            "A scientific tool attempt returned ERROR; inspect input validation and tool status."
        )
    return {
        "grounded": True,
        "complete": False,
        "issues": issues,
        "selected_fact_ids": list(dict.fromkeys(ids + warnings)),
        "selection_counts": {"required": len(required), "selected": len(required), "omitted": 0},
    }, decision
