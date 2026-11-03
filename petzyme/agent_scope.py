"""Scientific relevance at the agent boundary, using executed local results only."""

from __future__ import annotations

from collections.abc import Collection

from .agent_types import AgentState, ResearchRequest, ToolEvent
from .scientific_scope import (
    EvidenceScope,
    ScopeAssessment,
    assess_scientific_scope,
    interpret_scientific_scope,
)

# These native PAZy endpoint categories already carry source-reported values
# and units. Annotations and unrecognized/kinetic metrics cannot pass this gate.
RECORDED_PAZY_METRICS = frozenset({'specific_activity', 'product_concentration', 'conversion_percent'})


def assess_agent_scope(
    request: ResearchRequest, state: AgentState, selected: Collection[str] | None = None
) -> ScopeAssessment:
    """Flow: Receives original ResearchRequest and executed AgentState from validate_investigation(); returns ScopeAssessment from assess_scientific_scope() for independent relevance checks and scope facts.

    Source scope follows native tool contracts, never model claims, fact wording or names alone.
    PAZy annotations/embeddings are not endpoint or kinetic measurements.
    """
    evidence: EvidenceScope = {"materials": [], "quantities": [], "source_names": []}
    for event in state.events:
        if (
            event["status"] == "ERROR"
            or selected is not None
            and not set(event["fact_ids"]) & set(selected)
        ):
            continue
        result = state.results[event["call_id"]]
        value = result["result"]
        if value is None:
            continue
        if event['name'] == 'discover_pet_enzymes' and value['entries']:
            evidence['materials'].append('PET')
            evidence['quantities'].append('RECORDED_RESPONSE')
            evidence['source_names'].append('PAZy')
        observations = []
        if event["name"] == "parse_nrel":
            observations = value["observations"]
        elif event["name"] == "filter_observations":
            observations = value
        elif event["name"] == "summarize_condition":
            observations = value["leading_responses"]
        elif event["name"] == "investigate_conditions":
            observations = [o for c in value["conditions"] for o in c["observations"]]
        elif event["name"] == "investigate_pet_evidence":
            observations = value["nrel"]["observations"]
        elif event["name"] == "advise_next_experiment":
            observations = [o for item in value['evidence_results'] if item['result'] for o in item['result']['nrel']['observations']]
        if observations:
            if "PET" not in evidence["materials"]:
                evidence["materials"].append("PET")
            if "RECORDED_RESPONSE" not in evidence["quantities"]:
                evidence["quantities"].append("RECORDED_RESPONSE")
            if "NREL" not in evidence["source_names"]:
                evidence["source_names"].append("NREL")
        if (event['name'] == 'investigate_pet_evidence' and value['pazy']['returned'] or
                event['name'] == 'advise_next_experiment' and any(item['result'] and item['result']['pazy']['returned'] for item in value['evidence_results'])):
            if "PET" not in evidence["materials"]:
                evidence["materials"].append("PET")
            if "PUBLICATION_REPORTED_MEASUREMENT" not in evidence["quantities"]:
                evidence["quantities"].append("PUBLICATION_REPORTED_MEASUREMENT")
            selections = ([value] if event['name'] == 'investigate_pet_evidence' else
                          [item['result'] for item in value['evidence_results'] if item['result']])
            if any(m['record'].get('metric_type') in RECORDED_PAZY_METRICS
                   and type(m['record'].get('value')) in (int, float)
                   and isinstance(m['record'].get('unit'), str) and m['record']['unit']
                   for selection in selections for card in selection['pazy']['records'] for m in card['measurements']):
                if 'RECORDED_RESPONSE' not in evidence['quantities']:
                    evidence['quantities'].append('RECORDED_RESPONSE')
            if "PAZy" not in evidence["source_names"]:
                evidence["source_names"].append("PAZy")
    # Explicit current material or quantity always wins; authenticated prior scope fills only unspecified fields.
    scope = interpret_scientific_scope(request["question"])
    context = request.get("context_scope")
    if context is not None:
        if scope["material"] == "UNSPECIFIED":
            scope["material"], scope["materials"] = context["material"], context["materials"].copy()
        if scope["quantity"] == "UNSPECIFIED":
            scope["quantity"] = context["quantity"]
    return assess_scientific_scope(scope, evidence)


def append_scope_facts(request: ResearchRequest, state: AgentState, event: ToolEvent) -> None:
    """Flow: Receives ResearchRequest/AgentState and actual ToolEvent from run_investigation(); returns None after attaching bounded source-grounded UNANSWERED scope facts for Muse and final reporting.

    Only explicit incompatibilities or methodological questions add facts; ordinary legacy
    source-scoped requests retain their evidence-ID contracts. No records/values are fabricated.
    """
    assessment = assess_agent_scope(request, state)
    scope = assessment["request"]
    planning = scope["purpose"] == "EXPERIMENT_PLANNING"
    if (
        not planning and (not assessment['issues'] or event['name'] not in (
            'investigate_pet_evidence','filter_observations','investigate_conditions',
            'advise_next_experiment','discover_pet_enzymes'))
        and scope["material"] in ("PET", "UNSPECIFIED")
        and scope["quantity"]
        in (
            "RECORDED_RESPONSE",
            "UNSPECIFIED",
        )
    ):
        return
    if planning and event["name"] != "filter_observations":
        return
    sources = state.results[event["call_id"]]["provenance"]
    if not sources:
        return
    explanations = assessment["explanations"].copy()
    if planning:
        missing = []
        if scope["material"] == "UNSPECIFIED" or not request["pet_forms"]:
            missing.append("intended substrate and PET form")
        if not request["temperatures"]:
            missing.append("available temperatures")
        if not request["ph_values"]:
            missing.append("pH limits or assay pH")
        if missing:
            explanations.append(
                "Planning clarification: Which "
                + ", ".join(missing)
                + " should guide this experiment, and what measured outcome do you want to optimize? "
                "Candidate conditions will remain HYPOTHETICAL until justified; no established optimum is inferred."
            )
    for text in explanations:
        if any(f["text"] == text for f in state.facts):
            continue
        identifier = event["call_id"] + ".scope" + str(len(event["fact_ids"]) + 1)
        state.facts.append(
            {
                "id": identifier,
                "call_id": event["call_id"],
                "kind": "UNANSWERED",
                "text": text,
                "evidence_category": "HYPOTHETICAL"
                if text.startswith("HYPOTHETICAL")
                else "COMPUTED",
                "provenance": sources,
                "ephemeral": event["ephemeral"],
            }
        )
        event["fact_ids"].append(identifier)
