"""Bounded semantic intent protocol; plans are metadata, never scientific evidence."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from .agent_request import _NREL_ONLY_PATTERNS, _PAZY_ONLY_PATTERNS, _explicit_source_only
from .agent_types import ResearchRequest
from .contracts import JSONValue

Objective = Literal[
    "enzyme_discovery",
    "multi_study",
    "experiment_advice",
    "sequence_similarity",
    "identity",
    "references",
]
OBJECTIVES = (
    "enzyme_discovery",
    "multi_study",
    "experiment_advice",
    "sequence_similarity",
    "identity",
    "references",
)


class ResearchPlan(TypedDict):
    objectives: list[Objective]
    enzyme_identifiers: list[str]


class PlanFunction(TypedDict):
    name: Literal["plan_research"]
    description: str
    parameters: dict[str, JSONValue]


class PlanTool(TypedDict):
    type: Literal["function"]
    function: PlanFunction


def needs_semantic_plan(request: ResearchRequest) -> bool:
    """Flow: Receives conservative ResearchRequest from interpret_request(); returns bool for run_investigation() to request flexible Muse intent metadata before exposing scientific tools."""
    if not request["objectives"] and request["scientific_scope"]["material"] == "PET":
        return True
    return (
        not request["objectives"]
        and (
            request["scientific_scope"]["material"] == "PET" or bool(request["enzyme_identifiers"])
        )
        and not re.search(
            r"\b(?:investigat\w*|reported|documented|evidence|recorded)\b",
            request["question"],
            re.I,
        )
    )


def plan_tool() -> PlanTool:
    """Flow: Receives no external input from run_investigation(); returns PlanTool for run_investigation() to request Muse intent metadata, with no activity claims."""
    return {
        "type": "function",
        "function": {
            "name": "plan_research",
            "description": "Interpret the researcher objective, including unfamiliar wording. Select compositional objectives and only enzyme labels literally supplied in the question. Do not invent measurements, conditions, identities or units. Broad high-response/condition screening uses enzyme_discovery (one comparable assay group); next-experiment objectives use experiment_advice. Python enforces source restrictions and original scientific scope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objectives": {
                        "type": "array",
                        "items": {"enum": list(OBJECTIVES)},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "enzyme_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                    },
                },
                "required": ["objectives", "enzyme_identifiers"],
                "additionalProperties": False,
            },
        },
    }


def validate_research_plan(candidate: object, request: ResearchRequest) -> ResearchRequest:
    """Flow: Receives an untrusted Muse plan object and original ResearchRequest from run_investigation(); returns ResearchRequest for select_tool_catalog() or raises ValueError, never accepting model-authored scientific input."""
    if not isinstance(candidate, dict) or set(candidate) != {"objectives", "enzyme_identifiers"}:
        raise ValueError("INVALID_RESEARCH_PLAN_FIELDS")
    goals, labels = candidate["objectives"], candidate["enzyme_identifiers"]
    if (
        not isinstance(goals, list)
        or not 1 <= len(goals) <= 3
        or any(type(g) is not str or g not in OBJECTIVES for g in goals)
        or len(set(goals)) != len(goals)
        or not isinstance(labels, list)
        or len(labels) > 3
        or any(type(n) is not str for n in labels)
        or len({n.casefold() for n in labels}) != len(labels)
        or any(
            type(n) is not str
            or not 1 <= len(n) <= 80
            or not re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", request["question"], re.I)
            for n in labels
        )
    ):
        raise ValueError("INVALID_RESEARCH_PLAN_VALUE")
    if any(
        n.casefold() not in {x.casefold() for x in labels} for n in request["enzyme_identifiers"]
    ):
        raise ValueError("RESEARCH_PLAN_DROPPED_REQUESTED_IDENTIFIER")
    updated: ResearchRequest = {
        **request,
        "objectives": goals.copy(),
        "enzyme_identifiers": labels.copy(),
    }
    if set(goals) & {"multi_study", "enzyme_discovery", "experiment_advice"}:
        updated["sources"] = (
            ["NREL"]
            if _explicit_source_only(request["question"], _NREL_ONLY_PATTERNS)
            else ["PAZy"]
            if _explicit_source_only(request["question"], _PAZY_ONLY_PATTERNS)
            else ["NREL", "PAZy"]
        )
        if "experiment_advice" in goals and not (
            request["temperatures"] and request["ph_values"] and request["pet_forms"]
        ):
            updated["objectives"].append("exploratory_planning")
        if (
            (labels or "experiment_advice" in goals)
            and "multi_study" not in goals
            and updated["sources"] != ["NREL"]
        ):
            updated["objectives"].append("multi_study")
    return updated
