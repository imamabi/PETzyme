"""Application dialogue entry points; scientific decisions remain in the existing agent loop."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Callable

from .agent import run_investigation
from .agent_request import _NREL_ONLY_PATTERNS, _PAZY_ONLY_PATTERNS, _explicit_source_only
from .agent_types import ReasoningBackend, ResearchReport, ToolUpdate
from .clarification import parse_clarification_context
from .contracts import ToolResult
from .decision_context import new_decision_session
from .dialogue_rules import dialogue_catalog as dialogue_catalog
from .dialogue_rules import guard_dialogue_tool as guard_dialogue_tool
from .dialogue_rules import required_dialogue_fact_ids as required_dialogue_fact_ids
from .dialogue_rules import validate_clarification as validate_clarification
from .dialogue_rules import validate_dialogue_context as validate_dialogue_context
from .dialogue_types import DialogueContext as DialogueContext
from .dialogue_types import DialogueInput as DialogueInput
from .privacy import identifier_safe as _label
from .protein_triage import triage_protein
from .runtime_config import RuntimeLimits, empty_usage
from .scientific_scope import (
    ScientificScopeError,
    validate_direct_dialogue_scope,
)


def prepare_dialogue_turn(payload: DialogueInput) -> tuple[str, DialogueContext, str | None]:
    """Parse one bounded dialogue turn into a safe question, context, and private sequence.

    Purpose: application boundary for live Muse dialogue. A null context
        starts a new dialogue via parse_clarification_context; otherwise
        the retained context is validated and the turn is classified as
        exact-conditions (validated against device limits), sequence
        (kept private, referenced only as turn:sequence), rephrase
        request, or rejection/answer. Raw turns, sequences, and provider
        history are never stored.
    Args:
        payload: DialogueInput with 'context' (None or DialogueContext),
            'text' (at most 16384 chars), and 'input_kind'.
    Returns:
        (question, context, sequence) tuple: the privacy-safe question
            string for run_investigation, the updated DialogueContext,
            and the private local sequence or None.
    Raises:
        ValueError: on malformed payloads, invalid retained context,
            out-of-bounds conditions, or unfocused requests.
    Limitations:
        Sequence detection is heuristic (FASTA header, "here is the
            sequence", long alpha runs); when in doubt the Sequence reply
            type should be used explicitly.
    Flow: Receives external DialogueInput from worker_main(); returns safe question str, DialogueContext and private local sequence str | None for investigate_with_backend()."""
    if (
        not isinstance(payload, dict)
        or not {"context", "text", "input_kind"}
        <= payload.keys()
        <= {"context", "text", "input_kind", "decision_support"}
        or ("decision_support" in payload and type(payload["decision_support"]) is not bool)
    ):
        raise ValueError("Invalid dialogue input")
    text, kind = payload["text"], payload["input_kind"]
    if (
        not isinstance(text, str)
        or len(text) > 16384
        or kind not in ("ANSWER", "SEQUENCE", "REFERENCE")
    ):
        raise ValueError("Invalid bounded dialogue turn")
    if kind == "ANSWER":
        validate_direct_dialogue_scope(text)
        if (_explicit_source_only(text,_NREL_ONLY_PATTERNS) or _explicit_source_only(text,_PAZY_ONLY_PATTERNS)):
            raise ScientificScopeError('Source-restricted questions use the general one-shot investigation. This retained two-enzyme decision workflow requires both sources; it will not silently ignore your source restriction.')
    if payload["context"] is None:
        base = parse_clarification_context(text)
        initial: dict[str, object] = {
            **base,
            "model_attempts": 0,
            "tool_attempts": 0,
            "assay_ph": None,
            "pet_form": None,
            "focus_label": None,
            "pending_intent": None,
            "scientific_status": "PARTIAL",
            "usage": empty_usage(),
        }
        if payload.get("decision_support"):
            initial["decision"] = new_decision_session(base["temperatures_c"])
        if not validate_dialogue_context(initial):
            raise ValueError("Unsupported initial context")
        state: DialogueContext = initial
        note = "Initial enzyme-selection objective. Identity resolution has not yet been performed."
        sequence = None
    else:
        if not validate_dialogue_context(payload["context"]):
            raise ValueError("Invalid retained context")
        state = deepcopy(payload["context"])
        if "decision_support" in payload and payload["decision_support"] != ("decision" in state):
            raise ValueError("Reset explicitly before changing decision mode")
        sequence = None
        raw = text.strip().replace("’", "'")
        sequence_turn = (
            kind == "SEQUENCE"
            or raw.startswith(">")
            or raw.lower().startswith("here is the sequence:")
            or bool(re.fullmatch(r"[A-Za-z\s]{20,}", raw) and re.search(r"[A-Za-z]{20,}", raw))
        )
        conditions = re.fullmatch(
            r"(?:Please investigate (.+?) at )?pH\s*(\d+(?:\.\d+)?)\s*[,;]?\s*(?:using\s+)?(crystalline PET powder|amorphous PET film)[.]?",
            raw,
            re.I,
        )
        restricted = re.fullmatch(
            r"I can only run experiments at (-?\d+(?:\.\d+)?)\s*°?C(?: now)?[.!]?", raw, re.I
        )
        if "decision" in state and restricted is not None:
            temperature = float(restricted[1])
            if temperature not in state["decision"]["original_temperatures_c"]:
                raise ValueError(
                    "New equipment constraint must select an originally declared temperature"
                )
            state["temperatures_c"] = [temperature]
            note = "Researcher explicitly restricted available temperature. Reassess previous hypothetical alternatives against this updated constraint; source observations are unchanged."
        elif conditions is not None:
            focus, ph, form = conditions.groups()
            if (
                float(ph) > state["maximum_ph"]
                or not 0 <= float(ph) <= 14
                or focus is not None
                and focus not in state["confirmed_labels"]
            ):
                raise ValueError(
                    "Requested conditions exceed limits or focus is not an established label"
                )
            state["assay_ph"] = float(ph)
            state["pet_form"] = (
                "CRYSTALLINE_POWDER"
                if form.casefold() == "crystalline pet powder"
                else "AMORPHOUS_FILM"
            )
            state["focus_label"] = focus
            note = "Researcher supplied exact requested assay conditions within the device bounds. These are not observed outcomes; the original two-enzyme decision remains unresolved."
        elif sequence_turn:
            sequence = re.sub(r"^here is the sequence:\s*", "", raw, flags=re.I)
            note = "New private sequence supplied locally. Review turn:sequence via triage_protein before deciding."
        elif (
            "rephrase" in raw.casefold()
            or "word" in raw.casefold()
            and "different" in raw.casefold()
        ):
            if not state["pending_question"]:
                raise ValueError("No validated pending question to rephrase")
            note = "Researcher requests different natural wording for the pending information request, without changing its scientific intent, facts or constraints."
        elif kind == "REFERENCE":
            note = "A reference was supplied; offline reference lookup is unavailable. Do not invent its identity."
        elif re.match(r"^no(?:\b|[,!.])", raw, re.I):
            explicit = re.fullmatch(r"no,?\s+I\s+(?:do\s+not|don't)\s+mean\s+(.+?)\.?", raw, re.I)
            candidate = explicit[1] if explicit else state["pending_candidate"]
            index = state["pending_index"]
            if explicit:
                if candidate in state["confirmed_labels"]:
                    index = state["confirmed_labels"].index(candidate)
                elif candidate in state["requested_labels"]:
                    index = state["requested_labels"].index(candidate)
                elif candidate == state["pending_candidate"]:
                    index = state["pending_index"]
                else:
                    unresolved = [
                        i for i, label in enumerate(state["confirmed_labels"]) if label is None
                    ]
                    if len(unresolved) == 1:
                        index = unresolved[0]
                    elif (
                        state["pending_intent"] in ("IDENTIFIER", "CONFIRM_CANDIDATE")
                        and state["pending_index"] in unresolved
                    ):
                        index = state["pending_index"]
                    else:
                        raise ValueError(
                            "Rejection target is ambiguous; identify the original label"
                        )
            if candidate is not None and index is not None:
                lookup = triage_protein(name=candidate)["result"]
                cards = lookup["candidates"] if lookup is not None else []
                if any(
                    c["source_name"] == "NREL"
                    and c["name"] == candidate
                    and "EXACT_NAME" in c["basis"]
                    for c in cards
                ):
                    if candidate not in state["rejected_labels"][index]:
                        if len(state["rejected_labels"][index]) >= 5:
                            raise ValueError("Rejection bound reached")
                        state["rejected_labels"][index].append(candidate)
                    if state["confirmed_labels"][index] == candidate:
                        state["confirmed_labels"][index] = None
            state["pending_candidate"] = None
            note = "Researcher rejected an identity. Preserve the original objective and request identifying material when needed."
        elif raw.casefold() in ("yes", "y") and state["pending_candidate"] is not None:
            i = state["pending_index"]
            name = state["pending_candidate"]
            if i is not None and name not in state["rejected_labels"][i]:
                state["confirmed_labels"][i] = name
            state["pending_candidate"] = None
            note = "Researcher confirmed the offered source label only; construct and function remain unverified."
        elif _label(raw):
            note = "Researcher supplied this additional identifier for the pending target: " + raw
        else:
            raise ValueError(
                "Use a bounded identifier, explicit rejection, reference or sequence reply"
            )
    # No raw turn, sequence or provider history is placed in this model question or retained state.
    if state["pending_index"] is None:
        state["pending_index"] = next(
            (i for i, x in enumerate(state["confirmed_labels"]) if x is None), None
        )
    state["pending_question"] = ""
    safe = {
        "objective": state["objective"],
        "requested_labels": state["requested_labels"],
        "confirmed_labels": state["confirmed_labels"],
        "rejected_labels": state["rejected_labels"],
        "available_temperatures": [f"{t:g}°C" for t in state["temperatures_c"]],
        "maximum_permitted_ph": state["maximum_ph"],
        "exact_assay_ph": state["assay_ph"],
        "pet_form": state["pet_form"],
        "focus_label": state["focus_label"],
        "adaptive_decision_requested": "decision" in state,
        "pending_intent": state["pending_intent"],
        "previous_validated_question": payload["context"]["pending_question"]
        if payload["context"]
        else "",
        "pending_target": None
        if state["pending_index"] is None
        else state["requested_labels"][state["pending_index"]],
    }
    return (
        "Researcher decision context (equipment bounds, not assay measurements): "
        + json.dumps(safe)
        + ". "
        + note,
        state,
        sequence,
    )


def run_dialogue_turn(
    payload: DialogueInput,
    backend: ReasoningBackend,
    *,
    csv_path: str,
    on_tool_event: Callable[[ToolUpdate], None] | None = None,
    runtime_limits: RuntimeLimits | None = None,
    automatic_runtime: bool = False,
) -> ToolResult[ResearchReport]:
    """Run one dialogue turn through the existing investigation loop.

    Purpose: thin application entry point: prepares the turn, then calls
        run_investigation with the dialogue context and private sequence
        attached. Each turn uses fresh provider history; attempt budgets
        accumulate across the dialogue via the context counters.
    Args:
        payload: DialogueInput for prepare_dialogue_turn().
        backend: ReasoningBackend (live Muse or mock).
        csv_path: trusted local NREL CSV path.
        on_tool_event: optional observer for ToolUpdates.
        runtime_limits: optional RuntimeLimits (defaults are 6 model /
            8 tool attempts per turn).
    Returns:
        ToolResult[ResearchReport] from run_investigation().
    Flow: Receives external DialogueInput and backend from investigate_with_backend(); returns ToolResult[ResearchReport] from run_investigation() without retaining private sequence input."""
    question, context, sequence = prepare_dialogue_turn(payload)
    return run_investigation(
        question,
        backend,
        csv_path=csv_path,
        on_tool_event=on_tool_event,
        dialogue_context=context,
        local_sequence=sequence,
        runtime_limits=runtime_limits,
        automatic_runtime=automatic_runtime,
    )
