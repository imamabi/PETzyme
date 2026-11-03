"""Allowlisted tool projections and canonical facts; no model-authored scientific claims."""

from __future__ import annotations

import json
from copy import deepcopy

from .agent_types import Fact, ToolEvent, ToolUpdate
from .contracts import JSONValue, Source, ToolResult
from .experimental import cell_locator
from .nrel_units import standardized_nrel_values
from .pet_presenter import preview_pet_evidence
from .privacy import identifier_safe, safe_public_text, tool_privacy


def safe_text(value: object) -> str:
    """Screen one untrusted value into bounded public text (see privacy.safe_public_text).

    Flow: Receives untrusted source object from project_evidence(); returns shared privacy-screened str for model/public projections."""
    return safe_public_text(value)


def project_evidence(
    call_id: str, name: str, result: ToolResult
) -> tuple[dict[str, JSONValue], list[Fact]]:
    """Project one tool result into an allowlisted preview plus canonical facts.

    Purpose: the privacy and grounding boundary between raw results and
        everything the model or UI may see. Each tool has an explicit
        projection: native records become short source-attributed fact
        sentences (EVIDENCE), caveats become LIMITATION facts, failures
        become UNANSWERED facts, and advisor outputs become PROPOSAL
        facts. Sequences, vectors, hashes, and full API responses never
        appear; every fact keeps its evidence category and provenance.
    Args:
        call_id: the executed call's ID (prefixes generated fact IDs).
        name: executed tool name (selects the projection).
        result: the native ToolResult.
    Returns:
        (preview, facts) tuple: the compact preview dict followed by
            the new canonical facts (duplicates by text are skipped).
    Limitations:
        Projections are lossy by design; full values stay in the
            investigation's private artifact memory only.
    Flow: Receives ToolResult from execute_tool_call(); returns preview dict and list[Fact] for run_investigation() and validate_investigation()."""
    value = result["result"]
    ephemeral = tool_privacy(name).persistence != "NREL_ONLY"
    facts: list[Fact] = []
    sources = result["provenance"]
    preview: dict[str, JSONValue] = {}

    def add(
        text: str,
        kind: str = "EVIDENCE",
        category: str | None = None,
        provenance: list[Source] | None = None,
    ) -> None:
        if any(fact["text"] == text for fact in facts):
            return
        facts.append(
            {
                "id": f"{call_id}.f{len(facts) + 1}",
                "call_id": call_id,
                "kind": kind,
                "text": text,
                "evidence_category": category,
                "provenance": sources if provenance is None else provenance,
                "ephemeral": ephemeral,
            }
        )

    def observed(observation: dict) -> None:
        add(
            f"{safe_text(observation['enzyme_identifier'])}: recorded NREL response {observation['original_text']} "
            f"at {observation['temperature_c']}°C, pH {observation['ph']}, {observation['pet_form']}; "
            f"units UNKNOWN/UNVERIFIED [record {observation['source_row']}, {observation['source_column']}].",
            category="MEASURED",
            provenance=[observation["source"]],
        )

    if name == "triage_protein" and value is not None:
        preview = {
            key: value[key]
            for key in (
                "identity_status",
                "sequence_validation",
                "sequence_length",
                "flagged_symbols",
                "total_candidates",
                "omitted_candidates",
                "alignment_performed",
            )
        }
        preview["unsupported_symbols_present"] = any(
            x not in "BJXZUO" for x in value["flagged_symbols"]
        )
        preview["flagged_symbols"] = [x for x in value["flagged_symbols"] if x in "BJXZUO"][:6]
        preview["nrel_candidates"] = [
            {
                "name": safe_text(c["name"]),
                "basis": c["basis"],
                "source_locator": c["source"]["locator"],
                "source_doi": c["source"]["doi"],
            }
            for c in value["candidates"]
            if c["source_name"] == "NREL"
        ]
        preview["other_source_candidates"] = sum(
            c["source_name"] != "NREL" for c in value["candidates"]
        )
        add(
            "Local identity triage: " + json.dumps(preview, sort_keys=True, allow_nan=False),
            category="COMPUTED",
            provenance=[c["source"] for c in value["candidates"] if c["source_name"] == "NREL"]
            or sources,
        )
        for limit in result["limitations"]:
            add(limit, "LIMITATION")
        return preview, facts
    if result["status"] == "ERROR" or value is None:
        add(
            f"{name} did not return usable evidence; no substitute result was generated.",
            "UNANSWERED",
        )
        return {"status": result["status"], "message": safe_text(result["message"])}, facts
    if name == 'discover_pet_enzymes':
        group = value['group']
        preview = {k: value[k] for k in ('answer', 'entries', 'omitted_enzymes', 'eligible_groups',
                                        'excluded_assays', 'nrel_unit_status', 'global_ranking_supported', 'persistence')}
        preview['group'] = {**group, 'protocol': {k: safe_text(v) if isinstance(v, str) else v
                                               for k, v in group['protocol'].items()}} if group else None
        preview['entries'] = [{k: safe_text(v) if isinstance(v, str) else v for k, v in e.items()}
                              for e in value['entries']]
        add(value['answer'], category='COMPUTED')
        if group:
            add('Selected documented assay group: ' + json.dumps(preview['group'], sort_keys=True), category='COMPUTED')
        for entry in preview['entries']:
            source = {**sources[0], 'locator': sources[0]['url'] + entry['assay_locator'], 'doi': entry['publication_doi']}
            add('Within-group enzyme discovery: ' + json.dumps(entry, sort_keys=True), category='COMPUTED', provenance=[source])
        for limitation in result['limitations']:
            add(limitation, 'LIMITATION', 'COMPUTED')
        return preview, facts
    if name == "advise_next_experiment":
        add(value["answer"], category="COMPUTED")
        preview = {
            "scientific_status": value["scientific_status"],
            "partial_reasons": value["partial_reasons"],
            "persistence": value["persistence"],
            "components": [],
        }
        components = [
            *([("investigate_conditions", value["condition_result"])] if value['condition_result']['result'] is not None else []),
            *[("investigate_pet_evidence", item) for item in value["evidence_results"]],
            ("recommend_next_experiments", value["plan_result"]),
        ]
        for child_name, child in components:
            child_preview, child_facts = project_evidence(call_id, child_name, child)
            preview["components"].append(
                {"name": child_name, "status": child["status"], "preview": child_preview}
            )
            for fact in child_facts:
                add(fact["text"], fact["kind"], fact["evidence_category"], fact["provenance"])
        for limit in result["limitations"]:
            add(limit, "LIMITATION")
        return preview, facts
    if name == "investigate_pet_evidence":
        preview = preview_pet_evidence(value, safe_text)
        add(value["answer"], category="COMPUTED")
        add(
            "Multi-study aggregate coverage: " + json.dumps(
                {**value["coverage"], 'working_unit_representations': [
                    {'record': o['source_row'], 'column': o['source_column'], 'representation': o.get('working_units'),
                     'row_reference_linkage': o.get('source_row_links', [])}
                    for o in preview['nrel']['observations']], **({"selected_enzyme": preview["experimental_coverage"]}
                 if preview.get("experimental_coverage") is not None else {})}, sort_keys=True),
            category="COMPUTED",
        )
        for o in value["nrel"]["observations"]:
            observed(o)
        for card in preview["pazy"]["records"]:
            source = {**sources[-1], "doi": card["publication_doi"], "locator": card["locator"]}
            add(
                "PAZy publication-derived assay: "
                + json.dumps(card, sort_keys=True, allow_nan=False),
                category="MEASURED",
                provenance=[source],
            )
        for c in value["comparisons"]:
            add("Cross-source comparability: " + json.dumps(c, sort_keys=True), category="COMPUTED")
        add(value["overlap_interpretation"], "LIMITATION", "COMPUTED")
        add(
            "Direct quantitative superiority and causal attribution are unsupported; no activity prediction is generated.",
            "UNANSWERED",
            "COMPUTED",
        )
        for gap in value["gaps"]:
            add(f"{gap['priority']}. {gap['code']}: {gap['requirement']}", "UNANSWERED", "COMPUTED")
        if result["status"] == "PARTIAL":
            add(
                "Investigation is PARTIAL: " + "; ".join(value["missing_work"]),
                "LIMITATION",
                "COMPUTED",
            )
    elif name == "load_nrel":
        preview = {"row_count": len(value["rows"]), "column_count": len(value["columns"])}
    elif name == "find_similar_nrel_proteins":
        preview = {
            "query_identifier": safe_text(value["query_identifier"]),
            "coverage": value["coverage"].copy(),
            "model": value["model"].copy(),
            "devices_used": value["devices_used"],
            "fallbacks": value["fallbacks"],
            "neighbors": [
                {
                    "identifier": safe_text(n["identifier"]),
                    "source_row": n["source_row"],
                    "cosine_similarity": n["cosine_similarity"],
                }
                for n in value["neighbors"]
            ],
        }
        add(
            f"ESM-2 representation search for {safe_text(value['query_identifier'])} [NREL record {value['query_source_row']}]: "
            f"coverage {value['coverage']}; model {value['model']}; devices {value['devices_used']}; fallbacks {value['fallbacks']}. "
            "This ranking covers only the bounded candidate window.",
            category="COMPUTED",
        )
        for neighbor in value["neighbors"]:
            add(
                f"{safe_text(neighbor['identifier'])} [NREL record {neighbor['source_row']}]: computed ESM-2 cosine similarity "
                f"{neighbor['cosine_similarity']} to {safe_text(value['query_identifier'])}; not measured activity.",
                category="COMPUTED",
                provenance=neighbor["provenance"],
            )
        add(
            "PET activity superiority, catalytic efficiency, thermal stability, structure and functional equivalence "
            "remain UNKNOWN from this sequence-representation comparison.",
            "UNANSWERED",
            "COMPUTED",
        )
        if not value["neighbors"]:
            add(
                "No usable neighbors were returned from this bounded window; no whole-dataset absence is established.",
                "UNANSWERED",
                "COMPUTED",
            )
    elif name == "recommend_next_experiments":
        preview = {
            "scientific_status": value["scientific_status"],
            "basis_kind": value["basis_kind"],
            "candidate_count": value["candidate_count"],
            "omitted_count": value["omitted_count"],
            "omitted_categories": value["omitted_categories"],
            "actions": [
                {
                    "action_id": c["action_id"],
                    "category": c["category"],
                    "priority": c["priority"],
                    "feasibility": c["feasibility"]["status"],
                }
                for c in value["recommendations"]
            ],
        }
        add(
            f"Advisor generated a PROPOSED_ONLY plan: {value['candidate_count']} candidates, "
            f"{len(value['recommendations'])} returned, {value['omitted_count']} omitted "
            f"({value['omitted_categories']}); no experiment has been performed.",
            category="COMPUTED",
        )
        for action in value["recommendations"]:
            # Canonical action text retains every planning field, without raw scientific artifacts.
            detail = {
                key: action[key]
                for key in (
                    "action_id",
                    "category",
                    "objective",
                    "proposed_action",
                    "required_measurements",
                    "target_conditions",
                    "controlled_variables",
                    "varied_variables",
                    "expected_information",
                    "prerequisites",
                    "feasibility",
                    "priority",
                    "ranking_rationale",
                    "limitations",
                )
            }
            detail["evidence_basis"] = [
                {
                    "category": b["category"],
                    "statement": b["statement"],
                    "csv_sha256": b["csv_sha256"],
                    "locators": [s["locator"] for s in b["sources"]],
                }
                for b in action["evidence_basis"]
            ]
            action_sources = []
            for basis in action["evidence_basis"]:
                for source in basis["sources"]:
                    if source not in action_sources:
                        action_sources.append(source)
            add(
                "PROPOSED (not observed): " + json.dumps(detail, sort_keys=True, allow_nan=False),
                "PROPOSAL",
                "HYPOTHETICAL",
                action_sources,
            )
    elif name == "parse_nrel":
        preview = {
            "observation_count": len(value["observations"]),
            "missing_cells": value["missing_cells"],
        }
    elif name == "filter_observations":
        returned = min(len(value), 5)
        preview = {
            "observation_count": len(value),
            "returned_observations": returned,
            "omitted_observations": len(value) - returned,
        }
        for observation in value[:5]:
            observed(observation)
        if len(value) > returned:
            add(
                f"Filtered NREL observation preview: {len(value)} matching cells; {returned} returned, "
                f"{len(value) - returned} omitted from reportable observations. Source-order preview, not activity ranking; "
                "the full filtered selection remains available to local assessment tools through its typed reference.",
                "LIMITATION",
                "COMPUTED",
            )
    elif name == "summarize_condition":
        preview = {
            key: value[key]
            for key in (
                "condition",
                "observation_count",
                "recorded_zeros",
                "positive_recorded_responses",
            )
        }
        add(
            f"Exact-condition NREL summary {value['condition']}: {value['observation_count']} observations, "
            f"{value['recorded_zeros']} recorded zeros and {value['positive_recorded_responses']} positive recorded responses; "
            "leading recorded values are ordered within this condition only.",
            category="COMPUTED",
        )
        for observation in value["leading_responses"]:
            observed(observation)
    elif name == "assess_evidence":
        preview = {
            "observation_count": value["observation_count"],
            "issue_codes": [i["code"] for i in value["issues"]],
        }
        for issue in value["issues"]:
            add(
                f"{issue['code']} ({issue['severity']}): {issue['explanation']}",
                "LIMITATION",
                "COMPUTED",
                issue["provenance"],
            )
    elif name == "check_comparability":
        preview = {
            key: value[key]
            for key in (
                "category",
                "missing_fields",
                "blocking_differences",
                "descriptive_within_column_supported",
                "quantitative_comparison_supported",
            )
        }
        add(
            f"Comparability: {value['category']}; quantitative support: {value['quantitative_comparison_supported']}.",
            category="COMPUTED",
        )
        numerical = value.get('numerical_comparison')
        if numerical is not None:
            preview['numerical_comparison'] = {
                key: safe_text(item) if isinstance(item, str) else item
                for key, item in numerical.items()
            }
            add('Descriptive comparison: ' + json.dumps(preview['numerical_comparison'], allow_nan=False)
                + '. No statistical significance or causal effect is established.', category='COMPUTED')
        if not value["quantitative_comparison_supported"]:
            add(
                "A direct quantitative performance comparison is unsupported by the supplied evidence.",
                "UNANSWERED",
                "COMPUTED",
            )
    elif name == "investigate_conditions":
        preview = {
            "enzyme_identifier": value["enzyme_identifier"],
            "changed_variables": value["changed_variables"],
            "comparability": value["comparability"]["category"],
            "direct_comparison_supported": value["direct_comparison_supported"],
            "causal_attribution_supported": value["causal_attribution_supported"],
            "conditions": [
                {"condition": c["condition"], "status": c["status"]} for c in value["conditions"]
            ],
        }
        for condition in value["conditions"]:
            if condition["status"] == "MISSING":
                add(
                    f"No exact NREL observation was found for {safe_text(value['enzyme_identifier'])} at "
                    f"{condition['condition']}; missing is not zero and no prediction is made.",
                    "UNANSWERED",
                    "COMPUTED",
                )
            for observation in condition["observations"]:
                observed(observation)
        add(
            f"Comparability: {value['comparability']['category']}; direct quantitative comparison supported: "
            f"{value['direct_comparison_supported']}; causal attribution supported: {value['causal_attribution_supported']}.",
            category="COMPUTED",
        )
        numeric = value['comparability'].get('numerical_comparison')
        if value['direct_comparison_supported'] and numeric is not None:
            preview['numerical_comparison'] = {
                key: safe_text(item) if isinstance(item, str) else item for key, item in numeric.items()
            }
            add('Descriptive comparison: ' + json.dumps(preview['numerical_comparison'], allow_nan=False)
                + '. No statistical significance or causal effect is established.', category='COMPUTED')
        for text in value["unsupported"]:
            add(text, "UNANSWERED", "COMPUTED")
        for issue in value["evidence"]["issues"]:
            add(
                f"{issue['code']}: {issue['explanation']}",
                "LIMITATION",
                "COMPUTED",
                issue["provenance"],
            )
    elif name == "identify_evidence_gaps":
        preview = {"requirement_codes": [gap["code"] for gap in value]}
        for gap in value:
            add(
                f"{gap['priority']}. {gap['code']}: {gap['requirement']} Targets: {gap['target_conditions']}",
                "UNANSWERED",
                "COMPUTED",
            )
    elif name in ("search_pazy_pet", "get_pazy_metadata"):
        records = value if isinstance(value, list) else [value]
        minimal = []
        for protein in records:
            record = {
                "id": protein["id"],
                "name": safe_text(protein["name"]) if protein["name"] is not None else None,
                "uniprot_accession": safe_text(protein["uniprot_accession"])
                if protein["uniprot_accession"] is not None
                else None,
                "organism": safe_text(protein["organism"]["scientific_name"])
                if protein["organism"] and protein["organism"]["scientific_name"] is not None
                else None,
                "substrates": [
                    safe_text(s["abbreviation"]) if s["abbreviation"] is not None else None
                    for s in protein["substrates"][:5]
                ]
                if protein["substrates"] is not None
                else None,
            }
            minimal.append(record)
            add(
                f"PAZy live protein {record['id']}: {record['name']}; organism {record['organism']}; "
                f"source-listed substrate associations {record['substrates']}."
            )
        preview = {"proteins": minimal, "record_count": len(minimal)}
    elif name == "get_associated_literature":
        preview = {"reference_counts": result.get("reference_counts"), "references": []}
        for reference in value:
            record = reference["record"]
            if record is None:
                add(
                    "An explicit PAZy reference remains unresolved; no publication was guessed.",
                    "UNANSWERED",
                )
            else:
                minimal = {
                    "id": record["id"],
                    "doi": safe_text(record["doi"]) if record["doi"] is not None else None,
                    "title": safe_text(record["title"]) if record["title"] is not None else None,
                }
                preview["references"].append(minimal)
                add(
                    f"Explicit PAZy reference {minimal['id']}: {minimal['title']}; DOI {minimal['doi']}."
                )
        add(
            f"PAZy reference counts: {result.get('reference_counts')}; returned associations only, not an exhaustive literature review.",
            "LIMITATION",
        )
    elif name == "select_nrel_protein":
        preview = {"enzyme_identifier": value["enzyme_identifier"]}
    elif name == "compare_identity":
        preview = {"identity": value["identity"]}
        add(
            f"Source identity check: {value['identity']}. Exact supplied-sequence identity is not functional equivalence.",
            category="COMPUTED",
        )
    for limitation in result["limitations"]:
        add(limitation, "LIMITATION", None)
    return preview, facts


def model_evidence(event: dict, facts: list[Fact]) -> dict:
    """Reduce one tool reply to the model-visible envelope.

    Purpose: wraps refs, preview, and facts with an explicit
        "untrusted_data": True marker and strips fact provenance to
        name/url/doi only, so the model treats retrieved content as
        data and never sees internal locators.
    Flow: Receives ToolEvent from execute_tool_call() and projected list[Fact]; returns reduced dict for run_investigation() model context."""
    return {
        "untrusted_data": True,
        "error_code": event.get("error_code"),
        "status": event["status"],
        "refs": event["refs"],
        "preview": event["preview"],
        "facts": [
            {
                "id": fact["id"],
                "kind": fact["kind"],
                "text": fact["text"],
                "evidence_category": fact["evidence_category"],
                "sources": [
                    {"name": s["name"], "url": s["url"], "doi": s["doi"]}
                    for s in fact["provenance"]
                ],
            }
            for fact in facts
        ],
    }


def public_tool_update(
    event: ToolEvent, result: ToolResult, model_requests: int, tool_attempts: int
) -> ToolUpdate:
    """Build the detached public ToolUpdate shown to UI observers per tool call.

    Purpose: presentation projection parallel to project_evidence:
        summarizes the call for progress display and accumulates the
        observations, conditions, comparison, similarity, plan, gaps,
        advice, and multi-study previews that present_investigation
        later assembles into the ResearchView. Transient tools expose
        aggregate previews only, never native records.
    Args:
        event: the executed ToolEvent.
        result: the native ToolResult.
        model_requests: backend attempts used so far.
        tool_attempts: tool attempts used so far.
    Returns:
        ToolUpdate with call_id, name, status, summary, attempt counts,
            a 'data' dict of display fragments, limitations, and
            provenance.
    Flow: Receives ToolEvent from execute_tool_call() and ToolResult/counts from run_investigation(); returns detached ToolUpdate for its optional presentation observer."""
    name, value = event["name"], result["result"]
    update: ToolUpdate = {
        "error_code": event.get("error_code"),
        "call_id": event["call_id"],
        "name": safe_text(name),
        "status": event["status"],
        "summary": "No public scientific result returned.",
        "model_requests": model_requests,
        "tool_attempts": tool_attempts,
        "data": {},
        "limitations": [],
        "provenance": [],
    }
    if event["status"] == "ERROR" and value is None:
        update["summary"] = ("Tool request did not complete; "
                             + str(event.get("error_code") or "SCIENTIFIC_TOOL_ERROR")
                             + ". No substitute evidence was generated.")
        if name == "triage_protein":
            update["data"]["identity"] = {}
        return deepcopy(update)
    if (
        name
        in (
            "triage_protein",
            "advise_next_experiment",
            "investigate_pet_evidence",
            "load_nrel",
            "parse_nrel",
            "investigate_conditions",
            "recommend_next_experiments",
            "filter_observations",
            "assess_evidence",
            "check_comparability",
            "identify_evidence_gaps",
        )
        and event["status"] != "ERROR"
    ):
        update["data"]["model_arguments"] = deepcopy(event["arguments"])
    if name == 'discover_pet_enzymes':
        if value is not None:
            preview, _ = project_evidence(event['call_id'], name, result)
            update['data']['discovery'] = preview
            update['summary'] = 'Computed a bounded within-assay ranking; no global superiority or unit conversion.'
        update['limitations'] = result['limitations'].copy()
        return deepcopy(update)
    if name == "triage_protein":
        update["data"]["model_arguments"] = {
            key: value
            for key, value in event["arguments"].items()
            if (key == "sequence_input_ref" and value == "turn:sequence")
            or (key == "name" and identifier_safe(value))
        }
        update["data"]["identity"] = deepcopy(event["preview"])
        update["summary"] = (
            "Local identity triage completed; source bytes, sequences and hashes withheld."
        )
        update["limitations"] = result["limitations"].copy()
        return deepcopy(update)
    if name == "advise_next_experiment":
        update["limitations"] = result["limitations"].copy()
        if value is None:
            return deepcopy(update)
        for child_name, child in [
            *([("investigate_conditions", value["condition_result"])] if value['condition_result']['result'] is not None else []),
            ("recommend_next_experiments", value["plan_result"]),
        ]:
            child_event = {
                **event,
                "name": child_name,
                "status": child["status"],
                "ephemeral": tool_privacy(child_name).persistence != "NREL_ONLY",
            }
            child_update = public_tool_update(child_event, child, model_requests, tool_attempts)
            update["data"].update(child_update["data"])
            update["limitations"].extend(child_update["limitations"])
            for source in child_update["provenance"]:
                if source not in update["provenance"]:
                    update["provenance"].append(source)
        update["data"]["gaps"] = value["gaps"]
        update["data"]["advice"] = {
            "answer": value["answer"],
            "partial_reasons": value["partial_reasons"],
        }
        update["data"]["multi_study_runs"] = [
            {
                "call_id": event["call_id"] + "_" + str(i),
                "preview": preview_pet_evidence(item["result"], safe_text),
            }
            for i, item in enumerate(value["evidence_results"])
            if item["result"] is not None
        ]
        update["summary"] = (
            "Composed exact local evidence and conditional proposals; PAZy context is transient."
        )
        return deepcopy(update)
    if name == "investigate_pet_evidence":
        if value is not None:
            update["data"]["multi_study"] = preview_pet_evidence(value, safe_text)
            update["summary"] = (
                "Completed a bounded local multi-study selection; PAZy previews are transient."
            )
        update["limitations"] = result["limitations"].copy()
        return deepcopy(update)
    if event["ephemeral"]:
        update["summary"] = "PAZy-derived record values withheld from the public interface."
        update["limitations"] = [
            "PAZy record values and derived identity results are transient and withheld."
        ]
        counts = result.get("reference_counts")
        if counts:
            update["data"]["reference_counts"] = counts.copy()
        return update
    update["limitations"] = result["limitations"].copy()
    update["provenance"] = [s for s in result["provenance"] if s["name"] in ("NREL", "ESM-2")]
    if event["status"] == "ERROR" or value is None:
        return deepcopy(update)

    def observations(values: list[dict]) -> None:
        representations = standardized_nrel_values(values[:10])
        update["data"]["observations"] = [
            {
                **{
                    key: o[key]
                    for key in (
                        "enzyme_identifier",
                        "original_text",
                        "temperature_c",
                        "ph",
                        "pet_form",
                        "source_row",
                        "source_column",
                        "unit_status",
                        "csv_sha256",
                        "evidence_category",
                    )
                },
                "source_locator": cell_locator(o),
                'working_units': representation,
            }
            for o, representation in zip(values[:10], representations)
        ]
        if len(values) > 10:
            update["limitations"].append(
                f"Public table bounded to 10 observations; {len(values) - 10} additional observations omitted."
            )

    if name == "load_nrel":
        update["summary"] = f"Loaded local NREL table: {len(value['rows'])} records."
    elif name == "parse_nrel":
        update["summary"] = (
            f"Parsed {len(value['observations'])} observations; {value['missing_cells']} missing cells."
        )
    elif name == "investigate_conditions":
        observations([o for c in value["conditions"] for o in c["observations"]])
        update["data"]["record_identity"] = value.get("record_identity")
        update["data"]["conditions"] = [
            {"condition": c["condition"], "status": c["status"]} for c in value["conditions"]
        ]
        update["data"]["comparison"] = {
            k: value[k] for k in ("direct_comparison_supported", "causal_attribution_supported")
        }
        update["data"]["comparison"]["category"] = value["comparability"]["category"]
        if value['direct_comparison_supported'] and value['comparability'].get('numerical_comparison'):
            update['data']['comparison']['numerical_comparison'] = value['comparability']['numerical_comparison']
        update["summary"] = f"Exact-condition investigation: {value['comparability']['category']}."
        update["limitations"].extend(
            value["unsupported"] + [i["explanation"] for i in value["evidence"]["issues"]]
        )
    elif name == "filter_observations":
        observations(value)
        update["summary"] = f"Selected {len(value)} exact observations."
    elif name == "summarize_condition":
        observations(value["leading_responses"])
        update["summary"] = f"Summarized one condition: {value['observation_count']} observations."
    elif name in ("assess_evidence", "check_comparability"):
        update["limitations"].extend(i["explanation"] for i in value["issues"])
        update["summary"] = "Completed a deterministic evidence assessment."
        if name == "check_comparability":
            update["data"]["comparison"] = {
                "category": value["category"],
                "direct_comparison_supported": value["quantitative_comparison_supported"],
                "causal_attribution_supported": False,
            }
            if value['quantitative_comparison_supported'] and value.get('numerical_comparison'):
                update['data']['comparison']['numerical_comparison'] = value['numerical_comparison']
    elif name == "identify_evidence_gaps":
        update["data"]["gaps"] = value
        update["summary"] = f"Identified {len(value)} verification or measurement requirements."
    elif name == "find_similar_nrel_proteins":
        update["data"]["similarity"] = {
            k: value[k]
            for k in (
                "query_identifier",
                "query_source_row",
                "csv_sha256",
                "model",
                "coverage",
                "neighbors",
                "devices_used",
                "fallbacks",
            )
        }
        update["summary"] = (
            f"Searched {value['coverage']['searched_candidates']} candidates; {value['coverage']['excluded_in_window']} excluded in window."
        )
    elif name == "recommend_next_experiments":
        update["data"]["plan"] = value
        update["limitations"].extend(
            limit for action in value["recommendations"] for limit in action["limitations"]
        )
        update["summary"] = (
            f"Returned {len(value['recommendations'])} proposed actions; {value['omitted_count']} omitted."
        )
    else:
        update["summary"] = (
            "Scientific function completed; no detailed public projection is available."
        )
    # Observer mutation must not change native results, model previews or agent decisions.
    return deepcopy(update)
