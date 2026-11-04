"""One approved local tool call, using typed references instead of model-supplied records."""

from __future__ import annotations

import hashlib
import json

from .agent_evidence import project_evidence
from .agent_request import BINDINGS
from .agent_types import AgentState, ToolEvent
from .contracts import ToolResult
from .dialogue_rules import guard_dialogue_tool, sync_identity_context
from .privacy import tool_privacy
from .results import outcome
from .tools import invoke_tool, validate_arguments


def _register(state: AgentState, call_id: str, name: str, result: ToolResult) -> dict[str, str]:
    """Register a tool result's reusable values as typed opaque refs.

    Purpose: stores native tables, observations, investigations, and
        proteins in this investigation's private artifact memory under
        'call_id:label' handles, and records seen protein IDs and
        sequences for downstream guards (detail-ID lineage, outbound
        screening). ERROR/None results register nothing.
    Flow: Receives AgentState, call_id, tool name and ToolResult from execute_tool_call(); returns the new refs dict for the ToolEvent."""
    refs = {}
    value = result["result"]
    if result["status"] == "ERROR" or value is None:
        return refs

    def save(label: str, kind: str, data: object) -> None:
        handle = f"{call_id}:{label}"
        state.artifacts[handle] = (kind, data)
        refs[label] = handle

    if name == "load_nrel":
        save("table", "table", value)
        if "sequence" in value["columns"]:
            index = value["columns"].index("sequence")
            state.sequences.update(
                "".join(row[index].split()).upper() for row in value["rows"] if row[index]
            )
    if name in ("parse_nrel", "filter_observations", "summarize_condition"):
        observations = (
            value["observations"]
            if name == "parse_nrel"
            else value["leading_responses"]
            if name == "summarize_condition"
            else value
        )
        save("observations", "observations", observations)
        for index, observation in enumerate(observations[:5], start=1):
            save(f"observation_{index}", "observation", observation)
    if name == "investigate_conditions":
        save("investigation", "investigation", value)
    if name == "find_similar_nrel_proteins":
        save("similarity", "similarity_search", value)
    if name == "select_nrel_protein":
        save("nrel_protein", "nrel_protein", value)
    if name == "get_pazy_metadata":
        save("protein", "pazy_protein", value)
    if name == "compare_identity":
        save(
            "verification",
            "protein_verification",
            {
                "nrel_identifier": value["nrel_identifier"],
                "pazy_protein_id": value["pazy_identifier"],
                "identity": value["identity"],
                "provenance": result["provenance"],
            },
        )
    if name == "search_pazy_pet":
        state.live_protein_ids.update(protein["id"] for protein in value)
    if name in ("get_pazy_metadata", "search_pazy_pet"):
        for protein in value if isinstance(value, list) else [value]:
            if protein["amino_acid_sequence"]:
                state.sequences.add("".join(protein["amino_acid_sequence"].split()).upper())
    return refs


def execute_tool_call(call: dict, state: AgentState, catalog: list[dict]) -> ToolEvent:
    """Execute one model-requested tool call with full validation and lineage guards.

    Purpose: the only path from a model tool call to real execution.
        Enforces, in order: the tool-attempt budget, unique call IDs,
        catalog membership plus JSON-schema arguments, opaque-ref
        resolution with type checking, dialogue guards, trusted csv_path
        injection, constraint anti-fabrication (model-supplied
        constraints must equal the externally supplied ones), PAZy
        detail-ID lineage, and duplicate-call blocking. Every outcome —
        including failures — becomes projected facts, so the model can
        never see raw records and failures never yield substitutes.
    Args:
        call: model tool-call dict with id/type/function fields.
        state: live AgentState (budgets, artifacts, lineage).
        catalog: model-facing definitions from select_tool_catalog().
    Returns:
        ToolEvent with call_id, name, echoed arguments, status, new
            refs, allowlisted preview, fact_ids, and the ephemeral flag.
    Limitations:
        Each invocation consumes one tool attempt even when it fails
            validation; duplicate calls are blocked without returning
            cached results.
    Flow: Receives approved catalog from select_tool_catalog() and model call dict from run_investigation(); returns ToolEvent for its next evidence-driven step."""
    call_id, name = call["id"], call["function"]["name"]
    arguments = {}
    result = None
    refs = {}
    error_code = 'TOOL_BUDGET'
    if state.tool_limit is not None and state.tool_attempts >= state.tool_limit:
        result = outcome("ERROR", None, [], [], None, "Tool budget exhausted")
    else:
        state.tool_attempts += 1
        try:
            error_code = 'DUPLICATE_CALL_ID'
            if call_id in state.call_ids:
                raise ValueError("Duplicate call identifier")
            state.call_ids.add(call_id)
            error_code = 'ARGUMENT_SCHEMA'
            arguments = json.loads(call["function"]["arguments"])
            definition = next(
                (tool["function"] for tool in catalog if tool["function"]["name"] == name), None
            )
            if definition is None or not validate_arguments(arguments, definition["parameters"]):
                raise ValueError("Tool is unapproved or its arguments are invalid")
            resolved = arguments.copy()
            error_code = 'REFERENCE_LINEAGE'
            for key, expected in BINDINGS.get(name, {}).items():
                ref = resolved.pop(key + "_ref", None)
                if ref is None:
                    continue
                kind, data = state.artifacts[ref]
                if kind not in (expected if isinstance(expected, tuple) else (expected,)):
                    raise ValueError("Reference type mismatch")
                resolved[key] = data
            if state.dialogue_context is not None:
                error_code = 'DIALOGUE_GUARD'
                guard_dialogue_tool(name, resolved, state)
            if name == "load_nrel":
                resolved["csv_path"] = state.csv_path
            if name in ("recommend_next_experiments", "advise_next_experiment"):
                error_code = 'RESEARCHER_CONSTRAINTS'
                supplied = resolved.pop("optional_constraints", None)
                if supplied is not None and supplied != state.researcher_constraints:
                    raise ValueError(
                        "Only externally supplied researcher constraints may be relayed"
                    )
                resolved["optional_constraints"] = state.researcher_constraints
                if name == 'advise_next_experiment':
                    supplied_goal = resolved.pop('planning_goal', None)
                    if supplied_goal is not None and supplied_goal != state.planning_goal:
                        raise ValueError('Only the researcher-derived planning goal may be relayed')
                    resolved['planning_goal'] = state.planning_goal
                    resolved['request_window'] = state.request_window
            if name == "get_pazy_metadata":
                error_code = 'REFERENCE_LINEAGE'
                if resolved["protein_id"] not in state.live_protein_ids:
                    raise ValueError("Protein ID must come from this investigation live search")
            signature = hashlib.sha256(
                (name + json.dumps(resolved, sort_keys=True, allow_nan=False)).encode()
            ).hexdigest()
            error_code = 'DUPLICATE_SCIENTIFIC_CALL'
            if signature in state.signatures:
                raise ValueError("Duplicate scientific call blocked; no cached result returned")
            state.signatures.add(signature)
            error_code = 'SOURCE_OR_SCIENTIFIC_TOOL_ERROR'
            result = invoke_tool(name, resolved)
            refs = _register(state, call_id, name, result)
        except (ValueError, KeyError, TypeError, StopIteration, RecursionError, OverflowError, OSError, RuntimeError, ArithmeticError):
            result = outcome(
                "ERROR",
                None,
                [],
                [],
                None,
                "Invalid, duplicate or unapproved tool call; no substitute evidence",
            )
    preview, facts = project_evidence(call_id, name, result)
    event: ToolEvent = {
        'error_code': error_code if result['status']=='ERROR' else None,
        "call_id": call_id,
        "name": name,
        "arguments": arguments if isinstance(arguments, dict) else {},
        "status": result["status"],
        "refs": refs,
        "preview": preview,
        "fact_ids": [fact["id"] for fact in facts],
        "ephemeral": tool_privacy(name).persistence != "NREL_ONLY",
    }
    state.results[call_id] = result
    state.events.append(event)
    state.facts.extend(facts)
    if name == "triage_protein" and state.dialogue_context is not None:
        sync_identity_context(state, call_id, arguments)
    return event
