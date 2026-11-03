"""Minimal session metadata; scientific records and private inputs stay transient."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal, TypedDict, cast
from uuid import uuid4

from .contracts import ExperimentalWindow
from .dialogue_types import DialogueContext
from .pet_retrieval import EvidenceQuery
from .privacy import question_contains_private_input

TurnKind = Literal["question", "private_input"]


class CohortQuery(TypedDict):
    """A source-bound query valid only during the active investigation."""

    label: str
    query: EvidenceQuery
    source_scope: Literal["BOTH", "PAZY_ONLY"]


def revalidate_cohort_queries(
    discovery_question: str, current_question: str, filter_question: str | None = None,
    pazy_only: bool = False, planning: bool = False, prefer_filter: bool = False,
    source_scope_override: bool | None = None,
) -> list[CohortQuery]:
    """Flow: Receives original discovery and current historical question from the research dispatcher; returns transient source-bound CohortQuery values for execute_tool_call()."""
    from .agent_request import _NREL_ONLY_PATTERNS, _explicit_source_only, interpret_request
    from .enzyme_discovery import discover_pet_enzymes

    original = interpret_request(discovery_question)
    current = interpret_request(current_question)
    filters = interpret_request(filter_question or discovery_question) if planning else (interpret_request(filter_question) if filter_question else current)
    if not planning and not prefer_filter and (current["temperatures"] or current["ph_values"] or current["pet_forms"]):
        filters = current
    if _explicit_source_only(current_question, _NREL_ONLY_PATTERNS):
        raise ValueError("NREL-only cohort retrieval requires separately verified source identities")
    if "enzyme_discovery" not in original["objectives"]:
        raise ValueError("Original turn is not an authenticated discovery request")
    if len(filters["temperatures"]) > 1 or len(filters["ph_values"]) > 1 or len(filters["pet_forms"]) > 1:
        raise ValueError("Specify one historical value for each condition")
    count = re.search(r"\b(\d+)\s+(?:enzymes|proteins|hydrolases)\b|\btop\s+(\d+)\b", discovery_question, re.IGNORECASE)
    limit = min(5, int(next(value for value in count.groups() if value is not None))) if count else 5
    discovery = discover_pet_enzymes(
        limit=limit,
        temperature_c=next(iter(original["temperatures"]), None),
        ph=next(iter(original["ph_values"]), None),
        pet_form=cast(Literal["CRYSTALLINE_POWDER", "AMORPHOUS_FILM"] | None, next(iter(original["pet_forms"]), None)),
        include_nrel_references="NREL" in original["sources"],
    )
    cohort = discovery["result"]
    if cohort is None or not cohort["entries"]:
        raise ValueError("Original source cohort could not be revalidated")
    if "NREL" in original["sources"] and not any(
        source.get("name") == "NREL" for source in discovery["provenance"]
    ):
        raise ValueError("The original NREL source cannot be authenticated; cohort replay stopped")
    queries = []
    for entry in cohort["entries"]:
        query: EvidenceQuery = {"protein_id": entry["source_protein_id"]}
        if filters["temperatures"]:
            query["temperature_c"] = filters["temperatures"][0]
        if filters["ph_values"]:
            query["ph"] = filters["ph_values"][0]
        if filters["pet_forms"]:
            form = filters["pet_forms"][0]
            query["form"] = "powder" if form == "CRYSTALLINE_POWDER" else "film"
            query["morphology"] = "crystalline" if form == "CRYSTALLINE_POWDER" else "amorphous"
        pazy_scope = (source_scope_override if source_scope_override is not None
                      else pazy_only or current["sources"] == ["PAZy"])
        scope: Literal["BOTH", "PAZY_ONLY"] = "PAZY_ONLY" if pazy_scope else "BOTH"
        queries.append(CohortQuery(label=entry["enzyme"], query=query, source_scope=scope))
    return queries


@dataclass(frozen=True)
class ResearchTurn:
    """A researcher-authored, privacy-screened history entry."""

    number: int
    kind: TurnKind
    text: str | None
    result_status: Literal["SUCCESS", "PARTIAL", "NO_DATA", "ERROR"] | None = None
    nrel_count: int | None = None
    pazy_count: int | None = None


@dataclass(frozen=True)
class ResearchSession:
    """Session metadata, excluding provider output and source-derived records."""

    session_id: str
    turns: tuple[ResearchTurn, ...]
    active_questions: tuple[str, ...] = ()
    discovery_replayable: bool = False
    cohort_origin: str | None = None
    cohort_filter: str | None = None
    cohort_compare_filters: tuple[str, str] | None = None
    cohort_compare_scopes: tuple[bool, bool] | None = None
    historical_question: str | None = None
    previous_historical_question: str | None = None
    historical_pazy_only: bool | None = None
    previous_historical_pazy_only: bool | None = None
    cohort_pazy_only: bool = False
    cohort_fingerprint: str | None = None
    cohort_window: ExperimentalWindow | None = None
    sequence_label: str | None = None
    pending_label: str | None = None
    pending_candidate: str | None = None
    rejected_candidates: tuple[str, ...] = ()
    objectives: tuple[str, ...] = ()
    enzyme_identifiers: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    temperatures_c: tuple[float, ...] = ()
    available_temperatures_c: tuple[float, ...] = ()
    ph_values: tuple[float, ...] = ()
    pet_forms: tuple[str, ...] = ()
    maximum_ph: float | None = None
    pending_need: Literal["CONSTRAINTS", "FOCUS"] | None = None
    dialogue_context: DialogueContext | None = None


def new_research_session() -> ResearchSession:
    """Flow: Receives an external new-investigation request; returns ResearchSession for append_research_turn()."""
    return ResearchSession(session_id=uuid4().hex, turns=())


def append_research_turn(session: ResearchSession, text: str) -> ResearchSession:
    """Flow: Receives ResearchSession and external researcher text; returns ResearchSession for a future research router without retaining private inputs."""
    if not isinstance(session, ResearchSession) or not isinstance(text, str):
        raise ValueError("Invalid research session or turn")
    if not text.strip() or len(text) > 16384:
        raise ValueError("A bounded, nonempty research turn is required")
    private = question_contains_private_input(text)
    turn = ResearchTurn(
        number=len(session.turns) + 1,
        kind="private_input" if private else "question",
        text=None if private else text.strip(),
    )
    return replace(session, turns=(*session.turns, turn))


def record_turn_status(
    session: ResearchSession, status: Literal["SUCCESS", "PARTIAL", "NO_DATA", "ERROR"],
    nrel_count: int | None = None, pazy_count: int | None = None,
) -> ResearchSession:
    """Flow: Receives ResearchSession and bounded execution metadata from main(); returns ResearchSession for its privacy-safe conversation history."""
    if not session.turns or status not in ("SUCCESS", "PARTIAL", "NO_DATA", "ERROR"):
        raise ValueError("Invalid turn status")
    if any(value is not None and (type(value) is not int or not 0 <= value <= 100000)
           for value in (nrel_count, pazy_count)):
        raise ValueError("Invalid source count")
    last = replace(session.turns[-1], result_status=status, nrel_count=nrel_count, pazy_count=pazy_count)
    return replace(session, turns=(*session.turns[:-1], last))


def prepare_research_turn(session: ResearchSession, text: str, *, source_followup: bool = False) -> tuple[ResearchSession, str, bool]:
    """Flow: Receives ResearchSession and external turn text; returns updated ResearchSession, str and bool for the shared investigation dispatcher."""
    updated = append_research_turn(session, text)
    if updated.turns[-1].kind != "question":
        raise ValueError("Private sequence input requires local identity triage.")
    followup = bool(session.active_questions and (session.pending_need or source_followup or re.search(
        r"\b(?:those|these|them|that|this|its|their|previous|what about|test next|next experiment|"
        r"compare with|compare another|can only|only run|equipment|device|maximum\s+ph|minimum\s+ph|cannot exceed|pazy(?:\s+only|-only)? evidence)\b",
        text, re.IGNORECASE,
    )))
    prior = session.active_questions
    active = (*prior, text.strip()) if followup else (text.strip(),)
    if not followup:
        updated = replace(updated, discovery_replayable=False, sequence_label=None,
                          pending_label=None, pending_candidate=None, rejected_candidates=(),
                          dialogue_context=None, pending_need=None, historical_question=None,
                          previous_historical_question=None, cohort_compare_filters=None, cohort_compare_scopes=None,
                          historical_pazy_only=None, previous_historical_pazy_only=None,
                          cohort_origin=None, cohort_filter=None, cohort_pazy_only=False,
                          cohort_fingerprint=None, cohort_window=None)
    question = text.strip()
    return replace(updated, active_questions=active), question, followup


def same_discovery_referents(previous: object, current: object) -> bool:
    """Flow: Receives transient discovery objects from executed tools; returns bool for the shared investigation dispatcher without retaining PAZy records."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    old_group, new_group = previous.get("group"), current.get("group")
    old_entries, new_entries = previous.get("entries"), current.get("entries")
    if not isinstance(old_group, dict) or not isinstance(new_group, dict):
        return False
    if not isinstance(old_entries, list) or not isinstance(new_entries, list) or not old_entries:
        return False
    if old_group != new_group:
        return False
    try:
        return [entry["source_protein_id"] for entry in old_entries] == [
            entry["source_protein_id"] for entry in new_entries
        ]
    except (KeyError, TypeError):
        return False
