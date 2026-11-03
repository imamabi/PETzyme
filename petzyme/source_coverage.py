"""Executed-source receipts and a bounded application coverage safeguard.

Receipts describe retrieval, not independent experiments or scientific support.
Native locators are used only in transient sets to avoid counting a record twice.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Literal, TypedDict

from .agent_types import AgentState, ResearchRequest
from .contracts import JSONValue, SourceReceipt, SourceStatus


class CoverageFunction(TypedDict):
    name: Literal["investigate_pet_evidence"]
    arguments: str


class CoverageCall(TypedDict):
    id: str
    type: Literal["function"]
    function: CoverageFunction


def source_receipts(request: ResearchRequest, state: AgentState) -> list[SourceReceipt]:
    """Flow: Receives ResearchRequest and actual AgentState from run_investigation(); returns list[SourceReceipt] for coverage_safeguard_call(), run_investigation() and present_investigation(), never model-authored verification."""
    found: dict[str, set[tuple[Any, ...]]] = {"NREL": set(), "PAZy": set()}
    attempted: dict[str, int] = {"NREL": 0, "PAZy": 0}
    failed: set[str] = set()

    def observations(values: Iterable[Mapping[str, Any]]) -> None:
        for o in values:
            if request["enzyme_identifiers"] and str(o["enzyme_identifier"]).casefold() not in {
                n.casefold() for n in request["enzyme_identifiers"]
            }:
                continue
            found["NREL"].add((o["source_row"], o["source_column"]))

    def landscape(value: dict[str, Any]) -> None:
        scope = value.get("source_scope", "BOTH")
        if scope != "PAZY_ONLY":
            attempted["NREL"] += 1
            observations(value["nrel"]["observations"])
        attempted["PAZy"] += 1
        for card in value["pazy"]["records"]:
            for measurement in card["measurements"]:
                found["PAZy"].add((card["locator"], measurement["locator"]))

    if state.decision_observations:
        # Native revalidation reads pinned cells; this is not a model-selected tool call.
        observations(state.decision_observations)
        attempted["NREL"] += 1

    for event in state.events:
        name = event["name"]
        native = state.results[event["call_id"]]
        value = native["result"]
        if event["status"] == "ERROR":
            if event.get("error_code") not in (None, "SOURCE_OR_SCIENTIFIC_TOOL_ERROR"):
                continue  # An invalid model call did not query or fail a source.
            if name in ("investigate_pet_evidence", "advise_next_experiment"):
                failed.update(request["sources"])
            elif name == "load_nrel":
                failed.add("NREL")
            continue
        if value is None:
            continue
        if name == "investigate_pet_evidence":
            landscape(value)
        elif name == "advise_next_experiment":
            for component in value["evidence_results"]:
                if component["result"] is not None:
                    landscape(component["result"])
        elif name == "discover_pet_enzymes":
            attempted["PAZy"] += 1
            if event["arguments"].get("include_nrel_references", True):
                attempted["NREL"] += 1
            for entry in value["entries"]:
                found["PAZy"].update(
                    (entry["assay_locator"], m["locator"]) for m in entry["measurements"]
                )
                found["NREL"].update(
                    (o["source_row"], o["source_column"]) for o in entry["nrel_references"]
                )
        elif name in (
            "parse_nrel",
            "filter_observations",
            "summarize_condition",
            "investigate_conditions",
        ):
            attempted["NREL"] += 1
            values = (
                value["observations"]
                if name == "parse_nrel"
                else value
                if name == "filter_observations"
                else value["leading_responses"]
                if name == "summarize_condition"
                else [o for c in value["conditions"] for o in c["observations"]]
            )
            observations(values)
    receipts: list[SourceReceipt] = []
    experimental = bool(
        set(request["objectives"])
        & {
            "multi_study",
            "enzyme_discovery",
            "exploratory_planning",
            "ranking",
            "comparison",
            "gaps",
            "reliability",
            "experiment_advice",
        }
        or request["temperatures"]
        or request["ph_values"]
        or request["pet_forms"]
    )
    for source in ("NREL", "PAZy"):
        status: SourceStatus
        if source not in request["sources"]:
            status, reason = "EXCLUDED", "Outside the requested source scope."
        elif attempted[source]:
            status, reason = (
                (
                    "FOUND",
                    "Original bounded records retrieved; counts are not independent experiments.",
                )
                if found[source]
                else (
                    "EMPTY",
                    "Queried, with no matching measured records returned under the executed filters.",
                )
            )
        elif source in failed:
            status, reason = (
                "UNAVAILABLE",
                "The executed source tool failed; no fallback source snapshot was invented.",
            )
        elif not experimental:
            status, reason = (
                "NOT_RELEVANT",
                "This workflow does not query activity measurements; sequence, representation and biological annotation evidence remain separate scientific results.",
            )
        else:
            status, reason = (
                "PENDING",
                (
                    "No fresh PAZy retrieval this turn; prior PAZy records are not retained or reauthenticated. Only pinned NREL references were revalidated."
                    if source == "PAZy" and state.decision_observations
                    else "Relevant source retrieval has not executed."
                ),
            )
        receipts.append(
            {
                "source": source,
                "status": status,
                "measurements": len(found[source]),
                "query_attempts": attempted[source],
                "reason": reason,
            }
        )
    return receipts


def coverage_safeguard_call(
    request: ResearchRequest, state: AgentState, approved_tool_names: Collection[str]
) -> CoverageCall | None:
    """Flow: Receives ResearchRequest, AgentState and approved catalog from run_investigation(); returns CoverageCall or None for run_investigation() before final selection, labelled as an application safeguard."""
    if not set(request["objectives"]) & {"multi_study", "enzyme_discovery", "exploratory_planning"}:
        return None
    if not any(r["status"] == "PENDING" for r in source_receipts(request, state)):
        return None
    if "investigate_pet_evidence" not in approved_tool_names:
        return None
    query: dict[str, JSONValue] = {}
    if len(request["enzyme_identifiers"]) == 1:
        query["protein_name"] = request["enzyme_identifiers"][0]
    for values, field in ((request["temperatures"], "temperature_c"), (request["ph_values"], "ph")):
        if len(values) == 1:
            query[field] = values[0]
    if len(request["pet_forms"]) == 1:
        query["form"] = "powder" if request["pet_forms"][0] == "CRYSTALLINE_POWDER" else "film"
        query["morphology"] = "crystalline" if query["form"] == "powder" else "amorphous"
    identifier = "application_coverage_" + str(state.tool_attempts)
    while identifier in state.call_ids:
        identifier += "_x"
    return {
        "id": identifier,
        "type": "function",
        "function": {
            "name": "investigate_pet_evidence",
            "arguments": json.dumps(
                {
                    "query": query,
                    "source_scope": "PAZY_ONLY" if request["sources"] == ["PAZy"] else "BOTH",
                }
            ),
        },
    }
