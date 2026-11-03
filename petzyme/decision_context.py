"""Bounded NREL-reference continuity; never retain PAZy records or source responses."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import TypedDict, TypeGuard, cast

from .advisor_candidates import CONTROLS
from .agent_evidence import model_evidence, project_evidence
from .agent_types import AgentState
from .contracts import JSONValue
from .decision import (
    CRITERIA,
    DESIGNS,
    MAX_RETAINED_CELLS,
    REQUIRED_CONTROLS,
    DecisionError,
    retrieved_observations,
)
from .decision_types import (
    DecisionSession,
    ExperimentalDecision,
    ExperimentOption,
    Preference,
    PublicFact,
)
from .experimental import valid_condition
from .observations import parse_nrel
from .protein_triage import load_authenticated_nrel_reference
from .results import outcome


class RetainedEvidence(TypedDict):
    facts: list[PublicFact]
    untrusted_data: bool


class PriorDecisionContext(TypedDict):
    previous_alternatives: list[dict[str, JSONValue]]
    previous_ranking: list[str]
    previous_rationale: list[Preference]
    retained_nrel_cell_count: int
    omitted_reference_count: int
    pazy_context_previously_queried_not_retained: bool
    omission_count_semantics: str


def new_decision_session(temperatures: list[float]) -> DecisionSession:
    """Flow: Receives explicit equipment list[float] from prepare_dialogue_turn(); returns empty DecisionSession for that active dialogue only."""
    return {
        "original_temperatures_c": temperatures.copy(),
        "nrel_references": [],
        "omitted_reference_count": 0,
        "previous_alternatives": [],
        "previous_ranking": [],
        "previous_rationale": [],
        "pazy_context_seen": False,
    }


def valid_decision_session(value: object) -> TypeGuard[DecisionSession]:
    """Flow: Receives external session object from validate_dialogue_context(); returns bool before prepare_dialogue_turn() may restore references or hypothetical choices."""
    if not isinstance(value, dict) or set(value) != set(DecisionSession.__annotations__):
        return False
    ts = value["original_temperatures_c"]
    if (
        not isinstance(ts, list)
        or not 1 <= len(ts) <= 8
        or any(type(t) not in (int, float) or not math.isfinite(t) or t < -273.15 for t in ts)
    ):
        return False
    if (
        type(value["omitted_reference_count"]) is not int
        or value["omitted_reference_count"] < 0
        or type(value["pazy_context_seen"]) is not bool
    ):
        return False
    refs = value["nrel_references"]
    if not isinstance(refs, list) or len(refs) > MAX_RETAINED_CELLS:
        return False
    for ref in refs:
        if (
            not isinstance(ref, dict)
            or set(ref) != {"csv_sha256", "source_row", "source_column"}
            or not isinstance(ref["csv_sha256"], str)
            or not re.fullmatch("[a-f0-9]{64}", ref["csv_sha256"])
            or type(ref["source_row"]) is not int
            or ref["source_row"] < 2
            or not isinstance(ref["source_column"], str)
            or len(ref["source_column"]) > 100
        ):
            return False
    options = value["previous_alternatives"]
    ranking = value["previous_ranking"]
    reasons = value["previous_rationale"]
    if (
        not isinstance(options, list)
        or len(options) > 3
        or not isinstance(ranking, list)
        or not isinstance(reasons, list)
        or len(reasons) > 6
    ):
        return False
    if not options:
        return not ranking and not reasons
    if len(options) < 2 or any(
        not isinstance(p, dict) or set(p) != set(ExperimentOption.__annotations__) for p in options
    ):
        return False
    table = load_authenticated_nrel_reference()
    if table is None:
        return False
    names = {r[table["columns"].index("")] for r in table["rows"]}
    ids = []
    for p in options:
        if (
            not isinstance(p["option_id"], str)
            or not re.fullmatch("P[1-9]", p["option_id"])
            or not isinstance(p["design"], str)
            or p["design"] not in DESIGNS
        ):
            return False
        for key in (
            "enzymes",
            "controlled_variables",
            "varied_variables",
            "controls",
            "evidence_ids",
        ):
            if not isinstance(p[key], list) or not all(isinstance(s, str) for s in p[key]):
                return False
        if p["evidence_ids"] or not set(p["enzymes"]) <= names or not 1 <= len(p["enzymes"]) <= 2:
            return False
        if (
            not set(p["controlled_variables"]) <= set(CONTROLS)
            or not set(p["varied_variables"]) <= set(CONTROLS) | {"control condition"}
            or set(p["controls"]) != REQUIRED_CONTROLS
        ):
            return False
        if (
            not isinstance(p["temperatures_c"], list)
            or not 1 <= len(p["temperatures_c"]) <= 2
            or any(type(t) not in (int, float) or t not in ts for t in p["temperatures_c"])
        ):
            return False
        if any(
            not valid_condition({"temperature_c": t, "ph": p["ph"], "pet_form": p["pet_form"]})
            for t in p["temperatures_c"]
        ):
            return False
        ids.append(p["option_id"])
    if (
        len(ids) != len(set(ids))
        or len(ranking) != len(ids)
        or any(not isinstance(i, str) for i in ranking)
        or set(ranking) != set(ids)
    ):
        return False
    for reason in reasons:
        if not isinstance(reason, dict) or set(reason) != {
            "preferred",
            "over",
            "criterion",
            "evidence_ids",
        }:
            return False
        if (
            any(not isinstance(reason[k], str) for k in ("preferred", "over", "criterion"))
            or reason["preferred"] not in ids
            or reason["over"] not in ids
            or reason["criterion"] not in CRITERIA
            or reason["evidence_ids"] != []
        ):
            return False
    return True


def restore_decision_evidence(state: AgentState) -> RetainedEvidence:
    """Flow: Receives validated active AgentState from run_investigation(); returns RetainedEvidence for run_investigation() after reauthenticating retained NREL cell locators from current source bytes."""
    context = state.dialogue_context
    if context is None or "decision" not in context or not context["decision"]["nrel_references"]:
        return {"facts": [], "untrusted_data": True}
    table = load_authenticated_nrel_reference()
    if table is None:
        raise DecisionError("RETAINED_NREL_SOURCE_UNAVAILABLE")
    parsed = parse_nrel(table)["result"]
    if parsed is None:
        raise DecisionError("RETAINED_NREL_SOURCE_INVALID")
    index = {
        (o["csv_sha256"], o["source_row"], o["source_column"]): o for o in parsed["observations"]
    }
    selected = []
    for ref in context["decision"]["nrel_references"]:
        o = index.get((ref["csv_sha256"], ref["source_row"], ref["source_column"]))
        if o is None or o["enzyme_identifier"] not in context["confirmed_labels"]:
            raise DecisionError("RETAINED_REFERENCE_UNVERIFIED_OR_IDENTITY_REOPENED")
        selected.append(o)
    state.decision_observations = selected
    # This is automatic reference revalidation, not a model-selected tool call or cached response.
    facts = []
    for position, observation in enumerate(selected):
        call_id = f"retained_nrel_{position}"
        _, item_facts = project_evidence(
            call_id,
            "filter_observations",
            outcome("SUCCESS", [observation], [table["source"]], [], "MEASURED"),
        )
        facts.extend(item_facts)
        state.call_ids.add(call_id)
    state.facts.extend(facts)
    return cast(
        RetainedEvidence,
        model_evidence(
            {
                "status": "SUCCESS",
                "refs": {},
                "preview": {"automatic_reference_revalidation": len(selected)},
            },
            facts,
        ),
    )


def retain_decision(state: AgentState, result: ExperimentalDecision) -> None:
    """Flow: Receives validated ExperimentalDecision from run_investigation(); returns None after retaining NREL locators and hypothetical choices, with every provider/source-result ID stripped."""
    context = state.dialogue_context
    assert context is not None and "decision" in context
    cells = retrieved_observations(state)
    session = context["decision"]
    session["nrel_references"] = [
        {
            "csv_sha256": o["csv_sha256"],
            "source_row": o["source_row"],
            "source_column": o["source_column"],
        }
        for o in cells[:MAX_RETAINED_CELLS]
    ]
    # Remember earlier omissions; this is a maximum recorded prefix omission, not a union cardinality.
    session["omitted_reference_count"] = max(
        session["omitted_reference_count"], len(cells) - MAX_RETAINED_CELLS, 0
    )
    session["previous_alternatives"] = [deepcopy(a["proposal"]) for a in result["alternatives"]]
    for p in session["previous_alternatives"]:
        p["evidence_ids"] = []
    session["previous_ranking"] = result["ranking"].copy()
    session["previous_rationale"] = deepcopy(result["rationale"])
    for reason in session["previous_rationale"]:
        reason["evidence_ids"] = []
    session["pazy_context_seen"] |= any(
        e["name"] in ("investigate_pet_evidence", "advise_next_experiment") for e in state.events
    )


def decision_model_context(session: DecisionSession) -> PriorDecisionContext:
    """Flow: Receives typed DecisionSession from prepare_dialogue_turn(); returns PriorDecisionContext for run_investigation() without native records or prior tool IDs; temperature arrays retain the same numeric contract as proposals."""
    proposals: list[dict[str, JSONValue]] = []
    for p in session["previous_alternatives"]:
        proposals.append(
            cast(
                dict[str, JSONValue],
                {
                    **p,
                    "evidence_category": "HYPOTHETICAL",
                },
            )
        )
    return {
        "previous_alternatives": proposals,
        "previous_ranking": session["previous_ranking"],
        "previous_rationale": session["previous_rationale"],
        "retained_nrel_cell_count": len(session["nrel_references"]),
        "omitted_reference_count": session["omitted_reference_count"],
        "omission_count_semantics": "Maximum recorded prefix omission; not whole-source coverage or union cardinality.",
        "pazy_context_previously_queried_not_retained": session["pazy_context_seen"],
    }
