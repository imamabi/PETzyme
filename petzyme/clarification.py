"""Deterministic, session-local clarification; no scientific result/history cache."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Literal

from .dialogue_types import ClarificationState as ClarificationState
from .dialogue_types import ClarificationTurn as ClarificationTurn
from .protein_sequences import AMBIGUOUS, CANONICAL
from .protein_triage import triage_protein


def _name(text: str) -> bool:
    return (
        bool(text) and len(text) <= 120 and text.isascii() and not any(c in text for c in "\r\n<>")
    )


def _question(state: ClarificationState) -> None:
    pending = next((i for i, label in enumerate(state["confirmed_labels"]) if label is None), None)
    state["pending_index"] = pending
    if pending is None:
        state["pending_candidate"] = None
        state["pending_question"] = (
            "Original decision context resumed. Reference labels are established, but a general two-enzyme "
            "experimental comparison has not been executed. No winning enzyme is established. Verify matched pH within your limit, "
            "PET form, protocol, units and uncertainty before comparing experimental responses."
        )
    elif state["pending_candidate"]:
        state["pending_question"] = (
            f"Did you mean the NREL label {state['pending_candidate']} for {state['requested_labels'][pending]}? Reply Yes or No."
        )
    else:
        state["pending_question"] = (
            f"Please provide an accession, publication DOI, exact local name or FASTA sequence for "
            f"{state['requested_labels'][pending]}. No identity substitution has been made."
        )


def _resolve(
    state: ClarificationState, index: int, *, name: str | None = None, sequence: str | None = None
) -> tuple[str, str | None, str]:
    """Project triage to minimal labels only; never retain native records, headers or sequences."""
    result = triage_protein(name=name, sequence_input=sequence, limit=5)
    value = result["result"]
    outcome, validation = value["identity_status"], value["sequence_validation"]

    def feedback(message: str) -> tuple[str, str | None, str]:
        # This projection is displayed for one turn and never enters ClarificationState.
        if sequence is not None:
            message = f"Sequence length: {value['sequence_length']} residues. " + message
        return outcome, validation, message

    rejected = state["rejected_labels"][index]
    cards = value["candidates"]
    exact_sequence = {
        c["identifier"]
        for c in cards
        if c["source_name"] == "NREL" and "EXACT_SEQUENCE" in c["basis"]
    }
    # A sequence establishes a label only when the match is unique, complete (nothing omitted), and unrejected.
    if (
        sequence is not None
        and outcome == "IDENTIFIED_SEQUENCE"
        and len(exact_sequence) == 1
        and not value["omitted_candidates"]
    ):
        label = next(iter(exact_sequence))
        if label not in rejected:
            state["confirmed_labels"][index] = label
            source = next(
                c["source"]
                for c in cards
                if c["source_name"] == "NREL" and "EXACT_SEQUENCE" in c["basis"]
            )
            return feedback(
                "Exact checksum-bound NREL sequence match; construct identity and function remain unverified. "
                f"Source: {source['doi']}; {source['locator']}. {source['attribution']}"
            )
    if name is not None and any(c["name"] == name and "EXACT_NAME" in c["basis"] for c in cards):
        if name not in rejected:
            # This is the researcher's supplied source label, never a PAZy record/identifier cache.
            state["confirmed_labels"][index] = name
            return feedback(
                "Requested name is source-listed; this establishes a label, not sequence or construct identity."
            )
    choices = {
        c["identifier"]
        for c in cards
        if c["source_name"] == "NREL" and c["identifier"] not in rejected
    }
    if (
        sequence is None
        and len(choices) == 1
        and not value["omitted_candidates"]
        and len(rejected) < 5
    ):
        state["pending_candidate"] = next(iter(choices))
        return feedback("One source-backed NREL name candidate requires your confirmation.")
    return feedback(
        "Identity remains unresolved in this context. Previously rejected labels will not be suggested again. "
        "Non-NREL aliases and ambiguous/fragment matches require further verification; no PAZy selection is retained."
    )


def parse_clarification_context(question: str) -> ClarificationState:
    """Parse a bounded two-enzyme choice question into retained context.

    Purpose: extracts the CHOOSE_BETWEEN_ENZYMES objective, two distinct
        labels, one available-temperature list, and one maximum pH using
        an explicit grammar ("Should I use A or B? ... supports 40C and
        60C, ... maximum pH 7"). Unsupported phrasing is rejected rather
        than partially parsed, and sequence-like labels are refused here
        (sequences belong in the transient sequence reply form).
    Args:
        question: researcher question string, at most 2000 chars.
    Returns:
        ClarificationState with objective, requested/confirmed labels,
            rejected labels, temperatures, maximum_ph, and pending fields.
            No scientific resolution is performed.
    Raises:
        ValueError: on unparseable questions, missing/conflicting
            constraints, duplicate labels, or sequence-like input.
    Example:
        >>> state = parse_clarification_context(
        ...     'Should I use ESM050 or LCCMC? My device supports 40C and 60C, maximum pH 7.')
        >>> (state['requested_labels'], state['temperatures_c'], state['maximum_ph'])
        (['ESM050', 'LCCMC'], [40.0, 60.0], 7.0)
    Limitations:
        Bounded grammar, not general NLU; a maximum pH is an upper bound,
            never an assay setpoint.
    Flow: Receives external equipment question str from start_clarification() or prepare_dialogue_turn(); returns ClarificationState without scientific resolution for those callers."""
    if not isinstance(question, str) or len(question) > 2000:
        raise ValueError(
            "Use one bounded two-enzyme question with explicit temperatures and a maximum pH."
        )
    match = re.search(
        r"(?:(?:should\s+I\s+use|choose\s+between|compare)\s+|^\s*)(.{1,120}?)\s+(?:or|and|versus)\s+(.{1,120}?)(?=[?;,])",
        question,
        re.I,
    )
    if not match:
        raise ValueError(
            "Use “Should I use NAME or NAME?” followed by available temperatures and a maximum pH."
        )
    labels = [s.strip() for s in match.groups()]
    if not all(_name(label) for label in labels) or labels[0] == labels[1]:
        raise ValueError(
            "Supply two distinct bounded enzyme labels; sequence input belongs in the sequence reply form."
        )
    for label in labels:
        if len(label) >= 20 and set(label.upper()) <= CANONICAL | AMBIGUOUS:
            known = triage_protein(name=label)["result"]["candidates"]
            if not any(c["name"] == label and "EXACT_NAME" in c["basis"] for c in known):
                raise ValueError(
                    "Sequence-like input belongs in the transient sequence reply form, not the retained objective."
                )
    equipment = question[match.end() :]
    temperature = r"-?\d+(?:\.\d+)?\s*°?\s*C\b"
    temperature_list = "(" + temperature + r"(?:\s*(?:and|,|/)\s*" + temperature + r")*)"
    lists = list(
        re.finditer(
            r"(?:supports|\bcan\s+maintain|available temperatures(?: are)?)\s*:?\s*"
            + temperature_list
            + r"|\bI\s+have\s+"
            + temperature_list
            + r"\s+available\b",
            equipment,
            re.I,
        )
    )
    if len(lists) != 1:
        raise ValueError(
            "State one explicit available temperature list using “supports 40C and 60C”."
        )
    remaining = equipment[lists[0].end() :].lstrip()
    if remaining and remaining[0] not in ",;.?":
        raise ValueError(
            "Use an explicit C/°C list separated by and/comma/slash, followed by punctuation before other constraints."
        )
    temperatures = list(
        dict.fromkeys(
            float(t)
            for t in re.findall(
                r"(-?\d+(?:\.\d+)?)\s*°?\s*C\b",
                next(g for g in lists[0].groups() if g is not None),
                re.I,
            )
        )
    )
    maxima = [
        float(p)
        for pair in re.findall(
            r"(?:cannot|can't|can not|do not)\s+(?:(?:operate|work|run|go)\s+)?(?:above|over|exceed)\s+pH\s*(\d+(?:\.\d+)?)"
            r"|maximum\s+pH\s*(?:is\s+)?(\d+(?:\.\d+)?)|pH\s*(?:<=|≤)\s*(\d+(?:\.\d+)?)"
            r"|pH\s+cannot\s+exceed\s+(\d+(?:\.\d+)?)",
            equipment,
            re.I,
        )
        for p in pair
        if p
    ]
    if (
        not 1 <= len(temperatures) <= 8
        or any(not math.isfinite(t) or t < -273.15 for t in temperatures)
        or len(set(maxima)) != 1
        or not 0 <= maxima[0] <= 14
    ):
        raise ValueError(
            "Provide unambiguous available temperatures (C/°C) and one maximum pH; no constraints will be inferred."
        )
    state: ClarificationState = {
        "objective": "CHOOSE_BETWEEN_ENZYMES",
        "requested_labels": labels,
        "confirmed_labels": [None, None],
        "rejected_labels": [[], []],
        "temperatures_c": temperatures,
        "maximum_ph": maxima[0],
        "pending_index": None,
        "pending_candidate": None,
        "pending_question": "",
    }
    return state


def start_clarification(question: str) -> ClarificationTurn:
    """Start offline clarification: parse the question, resolve source-listed labels.

    Purpose: builds the initial ClarificationState, acknowledges labels
        that are already source-listed, and focuses the first unresolved
        identity with exactly one question. Deterministic local control;
        no model is involved.
    Args:
        question: bounded two-enzyme choice question (see
            parse_clarification_context).
    Returns:
        ClarificationTurn with 'state', 'message', and null
            triage_outcome/sequence_validation on the first turn.
    Raises:
        ValueError: propagated from parse_clarification_context on
            unparseable input.
    Flow: Receives external question str from submit_start(); returns ClarificationTurn via parse_clarification_context() for transient local UI and advance_clarification()."""
    state = parse_clarification_context(question)
    labels = state["requested_labels"]
    # Resolve exact source-listed labels before focusing one unresolved identity at a time.
    for i, label in enumerate(labels):
        _resolve(state, i, name=label)
        state["pending_candidate"] = None
    pending = next((i for i, label in enumerate(state["confirmed_labels"]) if label is None), None)
    if pending is not None:
        _resolve(state, pending, name=labels[pending])
    _question(state)
    return {
        "state": state,
        "message": "Original enzyme-choice objective and explicit equipment constraints retained.",
        "triage_outcome": None,
        "sequence_validation": None,
    }


def advance_clarification(
    state: ClarificationState,
    reply: str,
    input_kind: Literal["ANSWER", "SEQUENCE", "REFERENCE"] = "ANSWER",
) -> ClarificationTurn:
    """Advance clarification by one typed researcher reply (Yes/No/sequence/name).

    Purpose: handles confirmation ("Yes" confirms the pending candidate;
        "No" records it as rejected and never suggests it again),
        SEQUENCE replies (runs real triage_protein; only a unique
        unrejected exact sequence match establishes a label), ANSWER name
        replies, and REFERENCE replies (acknowledged transiently, never
        resolved offline). The objective, original labels, and equipment
        limits are never overwritten by a reply.
    Args:
        state: current ClarificationState.
        reply: researcher reply string.
        input_kind: 'ANSWER', 'SEQUENCE', or 'REFERENCE'.
    Returns:
        ClarificationTurn with the updated state (a copy; the input is
            untouched), a display message, and any triage outcome.
    Raises:
        ValueError: on a non-string reply or unknown input kind.
    Limitations:
        Fragments, ambiguous sequences, and multiple matches stay
            unresolved; no winner or protocol is ever produced here.
    Flow: Receives ClarificationState from start_clarification() or advance_clarification() and external reply str from submit_reply(); returns ClarificationTurn for transient display and active-session context continuation."""
    updated = deepcopy(state)
    outcome = validation = None
    message = "Reply did not change established identifiers, objective or equipment constraints."
    index = updated["pending_index"]
    if not isinstance(reply, str) or input_kind not in ("ANSWER", "SEQUENCE", "REFERENCE"):
        raise ValueError("Supply one typed reply.")
    text = reply.strip()
    if index is None:
        message = "Decision context is already resumed; reset explicitly to change identifiers or constraints. No enzyme winner is established."
    elif (
        text.lower() in ("no", "n") or re.match(r"^no\s*[,!.]", text, re.I)
    ) and input_kind == "ANSWER":
        candidate = updated["pending_candidate"]
        if candidate and candidate not in updated["rejected_labels"][index]:
            updated["rejected_labels"][index].append(candidate)
        updated["pending_candidate"] = None
        message = "Suggestion rejected. Original objective, other enzyme and equipment constraints are unchanged."
    elif text.lower() in ("yes", "y") and input_kind == "ANSWER" and updated["pending_candidate"]:
        updated["confirmed_labels"][index] = updated["pending_candidate"]
        updated["pending_candidate"] = None
        message = "Source label confirmed by the researcher; sequence, experimental construct and function remain unverified."
    elif input_kind == "SEQUENCE" or text.startswith(">"):
        updated["pending_candidate"] = None
        outcome, validation, message = _resolve(updated, index, sequence=reply)
    elif input_kind == "REFERENCE":
        message = "Reference received transiently; no accession/DOI lookup is performed offline. Provide a sequence or exact local name to proceed."
    elif _name(text) and text.lower() not in ("yes", "y"):
        updated["pending_candidate"] = None
        outcome, validation, message = _resolve(updated, index, name=text)
    if index is not None and updated["confirmed_labels"][index] is not None:
        next_index = next(
            (i for i, label in enumerate(updated["confirmed_labels"]) if label is None), None
        )
        if next_index is not None:
            _resolve(updated, next_index, name=updated["requested_labels"][next_index])
    _question(updated)
    return {
        "state": updated,
        "message": message,
        "triage_outcome": outcome,
        "sequence_validation": validation,
    }
