"""Display-ready public views over the existing bounded investigation workflow; no Streamlit dependency."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Callable, Literal, NotRequired, TypedDict, cast
from uuid import uuid4

from .agent import run_investigation
from .agent_evidence import project_evidence
from .agent_request import (
    _NREL_ONLY_PATTERNS,
    _PAZY_ONLY_PATTERNS,
    _explicit_source_only,
    contextualize_request,
    interpret_request,
)
from .agent_types import (
    PlanDiagnostics,
    ReasoningBackend,
    ResearchReport,
    ResearchRequest,
    RoutingTrace,
    ToolUpdate,
)
from .contracts import ExperimentalWindow, JSONValue, Source, SourceReceipt, ToolResult
from .dialogue import run_dialogue_turn
from .dialogue_rules import validate_dialogue_context
from .dialogue_types import DialogueContext, DialogueInput, DialogueReport
from .enzyme_discovery import EnzymeDiscovery, discover_pet_enzymes
from .meta import MetaModelClient
from .meta_telemetry import RequestTelemetry, accumulate_session_usage
from .pet_accumulation import MAX_RUNS, accumulate_pet_results
from .privacy import credential_like, question_contains_private_input, tool_privacy
from .protein_triage import IdentityTriage, triage_protein
from .research_session import (
    ResearchSession,
    append_research_turn,
    prepare_research_turn,
    same_discovery_referents,
)
from .results import outcome
from .runtime_config import RuntimeLimits, RuntimeLimitValues
from .scientific_scope import ScopeAssessment

ROOT = Path(__file__).resolve().parents[1]
_LOADED_AT = time.time()
_CODE_FINGERPRINT = hashlib.sha256(b''.join(p.read_bytes() for p in
    sorted((ROOT/'petzyme').glob('*.py'))) + (ROOT/'streamlit_app.py').read_bytes()).hexdigest()[:16]


class RuntimeIdentity(TypedDict):
    process_id: int
    loaded_at_epoch: float
    code_fingerprint: str


def runtime_identity() -> RuntimeIdentity:
    """Flow: Receives UI startup or present_investigation() invocation; returns immutable loaded-code RuntimeIdentity for Streamlit version/stale-worker diagnostics, without credentials or source data."""
    return {'process_id':os.getpid(),'loaded_at_epoch':_LOADED_AT,'code_fingerprint':_CODE_FINGERPRINT}

EXAMPLES = {
    "Temperature evidence": "Compare ESM050 activity evidence at 40°C and 60°C, pH 6.5, using crystalline PET powder.",
    "Missing conditions": "What evidence exists for ESM050 at 50°C and pH 8, and what information is missing?",
    "ESM-2 similarity": "Find proteins similar to LCC-ICCG using ESM-2.",
    "Next experiment": "What experiment should I conduct to determine whether ESM050 performs better at 60°C than at 40°C, at pH 6.5 using crystalline PET powder?",
    "Multi-study evidence": "What experimental PET-degradation evidence exists for LCC-ICCG across NREL and PAZy?",
    "Outside NREL evidence": "Which PAZy PET assays cover experimental conditions or publications outside the condensed NREL dataset?",
    "Multi-study comparison": "Can multi-study evidence across NREL and PAZy establish that ESM050 performs better at 60°C than 40°C, pH 6.5, using crystalline PET powder?",
    "Integrated experiment advisor": "What experiment should I perform next to determine whether ESM050's observed PET-degradation response differs meaningfully between 40°C and 60°C at pH 6.5 using crystalline PET powder? Use NREL and PAZy evidence.",
}


class HistoricalGroup(TypedDict):
    """One source-attributed group for transient side-by-side presentation."""

    question: str
    preview: dict[str, JSONValue]


class HistoricalAssessment(TypedDict):
    label: str
    category: str
    varying_variable: str | None
    missing_fields: list[str]
    blocking_differences: list[str]
    direct_comparison_supported: bool
    numerical_comparison: NotRequired[dict[str, JSONValue]]


class _TransientEvidence(TypedDict, total=False):
    persistence: Literal['NREL_ONLY','FORBIDDEN_PAZY_DERIVED']
    runtime: RuntimeIdentity
    interpretation: PlanDiagnostics
    source_coverage: list[SourceReceipt]
    routing: RoutingTrace
    discovery: EnzymeDiscovery
    run_id: str
    submitted_question: str
    planning_question: str
    scientific_scope: ScopeAssessment
    multi_study: dict[str, JSONValue]
    synthesis: dict[str, JSONValue]
    provider_telemetry: list[RequestTelemetry]
    execution_seconds: float
    selection_counts: dict[str, int]
    provider_budget: dict[str, JSONValue]
    advice: dict[str, JSONValue]
    communication: dict[str, JSONValue]
    dialogue: DialogueReport
    limits: RuntimeLimitValues
    cohort_missing: list[str]
    cohort_errors: list[str]
    restricted_annotation_status: list[str]
    comparison_groups: list[HistoricalGroup]
    group_assessments: list[HistoricalAssessment]


class ResearchView(_TransientEvidence):
    question: str
    mode: str
    workflow_status: str
    headline: str
    model_requests: int
    tool_attempts: int
    observations: list[dict[str, JSONValue]]
    conditions: list[dict[str, JSONValue]]
    comparison: dict[str, JSONValue] | None
    record_identity: dict[str, JSONValue] | None
    similarity: dict[str, JSONValue] | None
    plan: dict[str, JSONValue] | None
    gaps: list[dict[str, JSONValue]]
    limitations: list[str]
    issues: list[str]
    provenance: list[Source]
    progress: list[ToolUpdate]


class OfflineWorkflow:
    """Deterministic local backend over validated request objectives.

    Uses native tools and the same agent validators as live inference. Historical
    demonstrations retain their plans; other supported requests use local_research_plan().
    No model reasoning or arbitrary natural-language interpretation is claimed.
    """

    mode = "DETERMINISTIC_OFFLINE"

    def __init__(self, question: str, context_turns: list[str] | None = None) -> None:
        """Bind a validated local plan or historical demonstration.

        Raises ValueError when the scientific objective needs clarification.
        Flow: Receives external question str from run_offline(); initializes native tool decisions for complete().
        """
        self.request = contextualize_request(question, [] if context_turns is None else context_turns)
        self.local_landscape = ('multi_study' in self.request['objectives'] and len(self.request['enzyme_identifiers']) == 1
                                and 'experiment_advice' not in self.request['objectives']
                                and len(self.request['temperatures']) <= 1 and len(self.request['ph_values']) <= 1
                                and len(self.request['pet_forms']) <= 1)
        self.example = next((k for k, v in EXAMPLES.items() if v == question), 'Local research') if not context_turns else 'Local research'
        self.local_plan: list[tuple[str, dict]] | None = None
        if self.example == 'Local research':
            self.local_plan = local_research_plan(self.request)

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return the next scripted tool call, or the final fact-ID selection.

        Purpose: steps through the example's fixed plan (indexed by the
            number of tool replies so far), then selects all projected
            fact IDs — or the required checklist for integrated advice.
            Call IDs are u1, u2, ... with matching *_ref handles.
        Flow: Receives messages/tools from run_investigation(); returns assistant dict containing deterministic demonstration tool calls or canonical fact IDs."""
        replies = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        for message in messages:
            if message["role"] == "user" and message["content"].startswith('{"untrusted_data": true, "cohort_evidence"'):
                data = json.loads(message["content"])
                if data["planning"] and not replies:
                    return {"role": "assistant", "content": None, "tool_calls": [{"id": "cohort_advice", "type": "function", "function": {"name": "advise_next_experiment", "arguments": json.dumps({"enzyme_identifier": data["cohort_labels"][0]})}}]}
                if data["planning"] and replies:
                    return {"role": "assistant", "content": json.dumps({"fact_ids": replies[-1]["final_selection"]["required_fact_ids"]})}
                return {"role": "assistant", "content": json.dumps({"fact_ids": data["required_fact_ids"]})}
        a = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
        b = {**a, "temperature_c": 60}
        if self.example == "Missing conditions":
            a = {**a, "temperature_c": 50, "ph": 8}
            b = {**a, "pet_form": "AMORPHOUS_FILM"}
        plan = [
            ("load_nrel", {}),
            ("parse_nrel", {"table_ref": "u1:table"}),
            (
                "investigate_conditions",
                {
                    "observations_ref": "u2:observations",
                    "enzyme_identifier": "ESM050",
                    "condition_a": a,
                    "condition_b": b,
                },
            ),
        ]
        if self.local_plan is not None:
            plan = self.local_plan.copy()
            if ({'enzyme_discovery', 'experiment_advice'} <= set(self.request['objectives'])
                    and replies and replies[0].get('preview', {}).get('entries')):
                # Deterministic local continuation over a real executed source entry,
                # not a global winner or a scripted live model choice.
                label = replies[0]['preview']['entries'][0]['enzyme']
                plan.append(('advise_next_experiment', {'enzyme_identifier': label,
                    'source_scope': 'PAZY_ONLY' if self.request['sources'] == ['PAZy'] else 'BOTH'}))
        elif self.example == 'Local landscape':
            query = {'protein_name': self.request['enzyme_identifiers'][0]}
            for field, values in (('temperature_c', self.request['temperatures']), ('ph', self.request['ph_values'])):
                if values:
                    query[field] = values[0]
            if self.request['pet_forms']:
                query['form'] = 'powder' if self.request['pet_forms'][0] == 'CRYSTALLINE_POWDER' else 'film'
                query['morphology'] = 'crystalline' if self.request['pet_forms'][0] == 'CRYSTALLINE_POWDER' else 'amorphous'
            plan = [('investigate_pet_evidence', {'query': query, 'limit': 3})]
        elif self.example == "Integrated experiment advisor":
            plan = [
                (
                    "advise_next_experiment",
                    {"enzyme_identifier": "ESM050", "condition_a": a, "condition_b": b},
                )
            ]
        elif self.example == "Multi-study evidence":
            plan = [
                ("investigate_pet_evidence", {"query": {"protein_name": "LCC-ICCG"}, "limit": 3})
            ]
        elif self.example == "Outside NREL evidence":
            plan = [("investigate_pet_evidence", {"scope": "OUTSIDE_NREL_PUBLICATION", "limit": 3})]
        elif self.example == "Multi-study comparison":
            plan = [
                (
                    "investigate_pet_evidence",
                    {"query": {"protein_name": "ESM050", "ph": 6.5, "form": "powder"}, "limit": 3},
                )
            ]
        elif self.example == "ESM-2 similarity":
            plan = [
                ("load_nrel", {}),
                (
                    "find_similar_nrel_proteins",
                    {
                        "table_ref": "u1:table",
                        "enzyme_identifier": "LCC-ICCG",
                        "top_k": 5,
                        "max_candidates": 8,
                    },
                ),
            ]
        elif self.example == "Missing conditions":
            plan.append(("identify_evidence_gaps", {"investigation_ref": "u3:investigation"}))
        elif self.example == "Next experiment":
            plan.append(
                (
                    "recommend_next_experiments",
                    {"investigation_ref": "u3:investigation", "max_recommendations": 10},
                )
            )
        index = len(replies)
        if index < len(plan):
            name, arguments = plan[index]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"u{index + 1}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        identifiers = (
            replies[-1]["final_selection"]["required_fact_ids"]
            if self.example == "Integrated experiment advisor"
            else [f["id"] for r in replies for f in r["facts"]]
        )
        return {"role": "assistant", "content": json.dumps({"fact_ids": identifiers})}


def local_research_plan(request: ResearchRequest) -> list[tuple[str, dict]]:
    """Flow: Receives validated ResearchRequest from OfflineWorkflow(); returns native tool-name/argument tuples for its deterministic execution through run_investigation(), without model inference, presets or invented conditions."""
    goals = request['objectives']
    if 'unsupported' in goals:
        raise ValueError('Prediction and simulation are unsupported; request measured evidence or conditional experiment planning.')
    labels = request['enzyme_identifiers']
    if 'sequence_similarity' in goals:
        if len(labels) != 1:
            raise ValueError('Provide exactly one authenticated NREL enzyme identifier for the bounded ESM-2 search.')
        return [('load_nrel', {}), ('find_similar_nrel_proteins', {
            'table_ref': 'u1:table', 'enzyme_identifier': labels[0], 'top_k': 5, 'max_candidates': 8,
        })]
    if 'enzyme_discovery' in goals and request['sources']!=['NREL']:
        if any(len(request[k]) > 1 for k in ('temperatures', 'ph_values', 'pet_forms')):
            raise ValueError('Choose one comparable discovery group at a time; do not pool conditions.')
        arguments = {field: request[key][0] for key, field in [('temperatures','temperature_c'), ('ph_values','ph'),('pet_forms','pet_form')] if request[key]}
        arguments['include_nrel_references'] = 'NREL' in request['sources']
        return [('discover_pet_enzymes', arguments)]
    if 'multi_study' in goals and request['sources']!=['NREL']:
        scope = 'PAZY_ONLY' if request['sources'] == ['PAZy'] else 'BOTH'
        if {'experiment_advice', 'exploratory_planning'} & set(goals) and len(labels) == 1:
            arguments = {'enzyme_identifier': labels[0], 'source_scope': scope}
            if (scope == 'BOTH' and len(request['temperatures']) == 2
                    and len(request['ph_values']) == len(request['pet_forms']) == 1):
                conditions = [{'temperature_c': t, 'ph': request['ph_values'][0],
                               'pet_form': request['pet_forms'][0]} for t in request['temperatures']]
                arguments.update(condition_a=conditions[0], condition_b=conditions[1])
            return [('advise_next_experiment', arguments)]
        queries = labels or [None]
        plan = []
        for label in queries[:3]:
            query = {'protein_name': label} if label else {}
            for key, field in [('ph_values','ph'),('temperatures','temperature_c')]:
                if len(request[key]) == 1:
                    query[field] = request[key][0]
            if len(request['pet_forms']) == 1:
                query['form'] = 'powder' if request['pet_forms'][0] == 'CRYSTALLINE_POWDER' else 'film'
                query['morphology'] = 'crystalline' if query['form'] == 'powder' else 'amorphous'
            plan.append(('investigate_pet_evidence', {'query':query,'source_scope':scope}))
        return plan
    if request['sources'] == ['NREL'] and len(labels) <= 1:
        args = {'observations_ref':'u2:observations'}
        if labels:
            args['enzyme_identifier'] = labels[0]
        args.update({field: request[key][0] for key,field in [('temperatures','temperature_c'),('ph_values','ph'),('pet_forms','pet_form')] if len(request[key])==1})
        return [('load_nrel',{}),('parse_nrel',{'table_ref':'u1:table'}),('filter_observations',args),('assess_evidence',{'observations_ref':'u3:observations'})]
    raise ValueError('Clarify an enzyme identifier or a PET experimental objective; no matching local investigation plan is available.')


def present_investigation(
    result: ToolResult[ResearchReport], updates: list[ToolUpdate],
    *, coverage_question: str | None = None, comparison_questions: tuple[str, str] | None = None,
    comparison_pazy_only: bool = False,
    comparison_scopes: tuple[bool, bool] | None = None,
) -> ToolResult[ResearchView]:
    """Assemble the display-ready ResearchView from a report plus its updates.

    Purpose: merges per-call ToolUpdate fragments (observations,
        conditions, comparison, similarity, plan, gaps, advice,
        multi-study runs) with the final report into one public view:
        question, mode, workflow status, headline, attempt counts, and
        deduplicated provenance/limitations/issues. No Streamlit
        dependency — pure data shaping the app renders.
    Args:
        result: ToolResult[ResearchReport] from run_investigation().
        updates: detached ToolUpdates collected during the run.
    Returns:
        ToolResult[ResearchView]; ERROR when the report itself is None.
    Flow: Receives ToolResult[ResearchReport] and detached ToolUpdate list from run_investigation(); returns ToolResult[ResearchView] for investigate_with_backend() and the Streamlit main()."""
    report = result["result"]
    if report is None:
        return outcome("ERROR", None, [], [], None, result["message"])
    observations, conditions, sources, limits, gaps = [], [], [], [], []
    comparison = similarity = plan = record_identity = None
    multi_study = None
    multi_study_runs = []
    grouped_runs: list[list[dict]] = [[], []]
    advice = None
    for update in updates:
        data = update["data"]
        for observation in data.get("observations", []):
            if observation not in observations:
                observations.append(observation)
        for condition in data.get("conditions", []):
            if condition not in conditions:
                conditions.append(condition)
        for source in update["provenance"]:
            if source not in sources:
                sources.append(source)
        limits.extend(update["limitations"])
        if "comparison" in data:
            comparison = data["comparison"]
        if "record_identity" in data:
            record_identity = data["record_identity"]
        if "similarity" in data:
            similarity = data["similarity"]
        if "plan" in data:
            plan = data["plan"]
        if "gaps" in data:
            gaps = data["gaps"]
        if "advice" in data:
            advice = data["advice"]
        multi_study_runs.extend(data.get("multi_study_runs", []))
        if "multi_study" in data:
            run = {"call_id": update["call_id"], "preview": data["multi_study"]}
            group_index = data.get("cohort_group")
            if comparison_questions is not None and type(group_index) is int and group_index in (0, 1):
                grouped_runs[group_index].append(run)
            else:
                multi_study_runs.append(run)
    limits.extend(report["limitations"])
    missing = [c for c in conditions if c["status"] == "MISSING"]
    if plan:
        headline = f"{len(plan['recommendations'])} prioritized verification and measurement proposals. No experiment has been performed."
    elif similarity:
        headline = (
            f"{len(similarity['neighbors'])} representation neighbors returned from "
            f"{similarity['coverage']['searched_candidates']} searched candidates. PET activity remains unknown."
        )
    elif missing:
        headline = (
            "No exact observations at the requested conditions."
            if not observations
            else "Some requested conditions lack exact observations."
        )
        headline += " Missing is not zero; no values are predicted."
    elif (
        observations
        and report["validation"].get("scientific_scope", {}).get("request", {}).get("purpose")
        == "EXPERIMENT_PLANNING"
    ):
        headline = (
            f"Retrieved {len(observations)} source-bound NREL observations for exploratory planning. "
            "Clarify the laboratory constraints and measurement objective before choosing candidate conditions; "
            "no experimentally established optimum is verified."
        )
    elif comparison and observations:
        headline = (
            "Both requested responses are recorded. A quantitative performance comparison remains unsupported."
            if not comparison["direct_comparison_supported"]
            else "The source assessment supports a documented quantitative comparison; causality is not established."
        )
    elif observations or multi_study_runs:
        parts = (
            [f"{len(observations)} source-bound NREL observations"]
            if observations
            else []
        )
        if multi_study_runs:
            parts.append(
                f"{len(multi_study_runs)} bounded multi-study preview"
                f"{'s' if len(multi_study_runs) != 1 else ''} with NREL references and PAZy assay cards"
            )
        headline = (
            "Retrieved " + " plus ".join(parts) + "; see the combined table. "
            "Units are unverified; missing is not zero and no comparison is established."
        )
    else:
        headline = (
            "No publishable structured evidence was returned; inspect the workflow limitations."
        )
    if report["status"] != "COMPLETE":
        headline = "Partial investigation. " + headline
    view: ResearchView = {
        'persistence': report['persistence'],
        'runtime': runtime_identity(),
        "question": report["question"]
        if report["persistence"] == "NREL_ONLY"
        else "PAZy investigation — record details withheld",
        "mode": report["provider_mode"],
        "workflow_status": report["status"],
        "headline": headline,
        "model_requests": report["model_requests"],
        "tool_attempts": report["tool_attempts"],
        "observations": observations,
        "conditions": conditions,
        "comparison": comparison,
        "record_identity": record_identity,
        "similarity": similarity,
        "plan": plan,
        "gaps": gaps,
        "limitations": list(dict.fromkeys(limits)),
        "issues": report["validation"]["issues"],
        "provenance": sources,
        "progress": updates,
    }
    if report.get("dialogue"):
        view["dialogue"] = report["dialogue"]
    annotation_tools = {"search_pazy_pet", "get_pazy_metadata", "get_associated_literature"}
    annotation_status = [f"{u['name']}: {u['status']}" for u in updates
                         if u['name'] in annotation_tools
                         and not tool_privacy(u['name']).communication_allowed]
    if annotation_status:
        view['restricted_annotation_status'] = list(dict.fromkeys(annotation_status))
    missing_cohort = [u["data"].get("cohort_label") for u in updates
                      if u["call_id"].startswith("cohort_") and u["data"].get("cohort_missing") is True]
    if missing_cohort and all(isinstance(label, str) for label in missing_cohort):
        view["cohort_missing"] = cast(list[str], missing_cohort)
    failed_cohort = [u["data"].get("cohort_label") for u in updates
                     if u["call_id"].startswith("cohort_") and u["data"].get("cohort_error") is True]
    if failed_cohort and all(isinstance(label, str) for label in failed_cohort):
        view["cohort_errors"] = cast(list[str], failed_cohort)
    if report.get("limits"):
        view["limits"] = report["limits"]
    if report.get('routing'):
        view['routing'] = report['routing'].copy()
    if report.get('interpretation'):
        view['interpretation'] = report['interpretation']
    if report.get('source_coverage'):
        view['source_coverage'] = report['source_coverage']
    if report.get("synthesis"):
        view["synthesis"] = report["synthesis"]
    if "scientific_scope" in report["validation"]:
        view["scientific_scope"] = report["validation"]["scientific_scope"]
    if "exploratory_planning" in interpret_request(report["question"])["objectives"]:
        for fact in report["facts"]:
            if fact["text"].startswith("Planning clarification:"):
                view["planning_question"] = fact["text"]
                break
    if report.get("communication"):
        view["communication"] = report["communication"]
    if "selection_counts" in report["validation"]:
        view["selection_counts"] = report["validation"]["selection_counts"].copy()
    if multi_study_runs:
        if len(multi_study_runs) > MAX_RUNS:
            view["limitations"].append(
                f"{len(multi_study_runs) - MAX_RUNS} executed multi-study component previews omitted from the display union; canonical selection still covers every executed required source fact."
            )
        coverage_request = interpret_request(coverage_question or report["question"])
        labels = [u["data"].get("cohort_label") for u in updates
                  if u["call_id"].startswith("cohort_") and u["data"].get("cohort_label") is not None]
        if labels and all(isinstance(label, str) for label in labels):
            coverage_request["enzyme_identifiers"] = cast(list[str], labels)
        multi_study = accumulate_pet_results(
            multi_study_runs[:MAX_RUNS], coverage_request, additional_nrel=observations
        )
    if multi_study:
        view["multi_study"] = multi_study
        view["headline"] = multi_study["answer"]
    if comparison_questions is not None:
        groups: list[HistoricalGroup] = []
        for index, group_question in enumerate(comparison_questions):
            runs = grouped_runs[index]
            if runs:
                group_request = interpret_request(group_question)
                if comparison_scopes[index] if comparison_scopes is not None else comparison_pazy_only:
                    group_request['sources'] = ['PAZy']
                labels = [u['data'].get('cohort_label') for u in updates
                          if u['data'].get('cohort_group') == index and u['data'].get('cohort_label')]
                if labels and all(isinstance(label, str) for label in labels):
                    group_request['enzyme_identifiers'] = cast(list[str], labels)
                preview = accumulate_pet_results(runs[:MAX_RUNS], group_request)
                groups.append({'question': group_question, 'preview': preview})
        view['comparison_groups'] = groups
        assessments = [cast(HistoricalAssessment, u['data']['group_assessment']) for u in updates
                       if isinstance(u['data'].get('group_assessment'), dict)]
        if assessments:
            view['group_assessments'] = assessments
        parsed_groups = [interpret_request(q) for q in comparison_questions]
        changed = [label for field, label in (('temperatures', 'temperature'), ('ph_values', 'pH'),
                                                ('pet_forms', 'PET form'))
                   if parsed_groups[0][field] != parsed_groups[1][field]]
        condition_note = (' Requested filters differ in ' + ', '.join(changed) + '.' if changed else '')
        source_note = (' Source restrictions differ between the two groups.'
                       if comparison_scopes is not None and comparison_scopes[0] != comparison_scopes[1] else '')
        assessment_note = ('NREL source-cell assessments are shown below; no PAZy construct equivalence is inferred.'
                           if assessments else 'A record-level protocol and identity assessment has not been run.')
        view['headline'] = ('The historical groups are displayed separately. Quantitative comparability '
                            'is not established. ' + assessment_note
                            + condition_note + source_note + ' NREL source unit declarations remain unverified where NREL is shown.')
    if advice:
        view["advice"] = advice
        view["headline"] = advice["answer"]
        if plan and 'planning_question' not in view:
            clarification = next((a for a in plan['recommendations'] if a.get('code') == 'DEFINE_LABORATORY_WINDOW'), None)
            if clarification is not None:
                view['planning_question'] = 'Planning clarification: ' + clarification['proposed_action']
    for update in updates:
        if 'discovery' in update['data']:
            view['discovery'] = cast(EnzymeDiscovery, update['data']['discovery'])
    if view.get('discovery') and report['provider_mode'] == 'DETERMINISTIC_OFFLINE':
        view['mode'] = 'DETERMINISTIC_LOCAL_DISCOVERY'
        view['model_requests'] = 0  # Local orchestration is not a model invocation.
    view['run_id'] = uuid4().hex[:12]
    if multi_study or advice or view.get('discovery'):
        view['submitted_question'] = report['question']
    return outcome(result["status"], view, sources, view["limitations"], "COMPUTED")


def investigate_with_backend(
    question: str,
    backend: ReasoningBackend,
    *,
    on_update: Callable[[ToolUpdate], None] | None = None,
    csv_path: str = str(ROOT / "data/nrel/label_data.csv"),
    dialogue_context: DialogueContext | None = None,
    local_sequence: str | None = None,
    dialogue_input: DialogueInput | None = None,
    runtime_limits: RuntimeLimits | None = None,
    automatic_runtime: bool = False,
    cohort_origin: str | None = None,
    cohort_filter: str | None = None,
    cohort_compare_filters: tuple[str, str] | None = None,
    cohort_compare_scopes: tuple[bool, bool] | None = None,
    cohort_pazy_only: bool = False,
    cohort_window: ExperimentalWindow | None = None,
    context_turns: list[str] | None = None,
) -> ToolResult[ResearchView]:
    """Run an investigation with any backend; present it with telemetry attached.

    Purpose: shared runner behind offline, mocked, and live modes.
        Collects ToolUpdates, times the run, routes dialogue payloads
        through run_dialogue_turn (all else through run_investigation),
        accumulates session usage into dialogue contexts, and attaches
        provider telemetry plus the spending-ceiling state for real
        MetaModelClient backends.
    Args:
        question: researcher question (used for non-dialogue runs).
        backend: ReasoningBackend (scripted, mock, or live).
        on_update: optional live progress observer.
        csv_path: trusted local NREL CSV path.
        dialogue_context: optional context for continuing dialogue runs.
        local_sequence: optional private sequence (dialogue only).
        dialogue_input: optional DialogueInput (takes the dialogue path).
        runtime_limits: optional RuntimeLimits.
    Returns:
        ToolResult[ResearchView] from present_investigation(), with
            execution_seconds and optional provider telemetry/budget.
    Flow: Receives external question str and ReasoningBackend from run_offline() or worker_main(); returns ToolResult[ResearchView] from present_investigation() for the Streamlit main()."""
    updates: list[ToolUpdate] = []

    def observe(update: ToolUpdate) -> None:
        updates.append(update)
        if on_update is not None:
            on_update(deepcopy(update))

    started = time.monotonic()
    telemetry_start = len(backend.telemetry) if isinstance(backend, MetaModelClient) else 0
    if dialogue_input is not None:
        result = run_dialogue_turn(
            dialogue_input,
            backend,
            csv_path=csv_path,
            on_tool_event=observe,
            runtime_limits=runtime_limits,
            automatic_runtime=automatic_runtime,
        )
    else:
        result = run_investigation(
            question,
            backend,
            csv_path=csv_path,
            on_tool_event=observe,
            dialogue_context=dialogue_context,
            local_sequence=local_sequence,
            runtime_limits=runtime_limits,
            automatic_runtime=automatic_runtime,
            cohort_origin=cohort_origin,
            cohort_filter=cohort_filter,
            cohort_compare_filters=cohort_compare_filters,
            cohort_compare_scopes=cohort_compare_scopes,
            cohort_pazy_only=cohort_pazy_only,
            cohort_window=cohort_window,
            context_turns=context_turns,
        )
    elapsed = round(time.monotonic() - started, 6)
    entries = (
        deepcopy(backend.telemetry[telemetry_start:])
        if isinstance(backend, MetaModelClient)
        else []
    )
    if result["result"] is not None and "dialogue" in result["result"]:
        context = result["result"]["dialogue"]["context"]
        context["usage"] = accumulate_session_usage(context["usage"], entries, elapsed)
    coverage_question = None
    if cohort_origin:
        current = interpret_request(question)
        coverage_question = (question if current["temperatures"] or current["ph_values"] or current["pet_forms"]
                             else cohort_filter or question)
    presented = present_investigation(result, updates, coverage_question=coverage_question,
                                       comparison_questions=cohort_compare_filters,
                                       comparison_pazy_only=cohort_pazy_only,
                                       comparison_scopes=cohort_compare_scopes)
    if presented["result"] is not None:
        presented["result"]["execution_seconds"] = elapsed
        if dialogue_input is not None:
            submitted = dialogue_input["text"]
            presented["result"]["submitted_question"] = (
                submitted if dialogue_input["input_kind"] == "ANSWER"
                and len(submitted) <= 4000 and not question_contains_private_input(submitted)
                else "Local identity clarification input (private sequence or reference withheld)."
            )
        # Only the real adapter's fixed numeric schema is accepted, never an arbitrary backend attribute.
        if isinstance(backend, MetaModelClient):
            presented["result"]["provider_telemetry"] = deepcopy(
                backend.telemetry[telemetry_start:]
            )
            presented["result"]["provider_budget"] = {
                "limit_usd": backend.max_estimated_cost_usd,
                "blocked": backend.budget_blocked,
            }
    return presented


def run_local_discovery(question: str) -> ToolResult[ResearchView]:
    """Flow: Receives external discovery question str from run_offline() or Streamlit main(); returns transient ToolResult[ResearchView] from discover_pet_enzymes() and project_evidence(), with a fresh run ID and zero model requests."""
    request = interpret_request(question)
    if 'enzyme_discovery' not in request['objectives']:
        return outcome('ERROR', None, [], [], None, 'This question is not a supported broad PET-enzyme discovery request.')
    if any(len(request[k]) > 1 for k in ('temperatures', 'ph_values', 'pet_forms')):
        return outcome('ERROR', None, [], [], None, 'Choose one exact condition group, or omit conditions for a documented-group discovery.')
    count = re.search(r'\b(\d+)\s+(?:enzymes|proteins|hydrolases)\b|\btop\s+(\d+)\b', question, re.I)
    limit = min(5, int(next(x for x in count.groups() if x is not None))) if count else 5
    result = discover_pet_enzymes(limit=limit,
        temperature_c=next(iter(request['temperatures']), None), ph=next(iter(request['ph_values']), None),
        pet_form=next(iter(request['pet_forms']), None))
    if result['result'] is None:
        return outcome('ERROR', None, result['provenance'], result['limitations'], None, result['message'])
    preview, _ = project_evidence('local_discovery', 'discover_pet_enzymes', result)
    view: ResearchView = {
        'persistence': 'FORBIDDEN_PAZY_DERIVED',
        'question': question, 'submitted_question': question, 'run_id': uuid4().hex[:12],
        'mode': 'DETERMINISTIC_LOCAL_DISCOVERY', 'workflow_status': result['status'],
        'headline': result['result']['answer'], 'model_requests': 0, 'tool_attempts': 1,
        'observations': [], 'conditions': [], 'comparison': None, 'record_identity': None,
        'similarity': None, 'plan': None, 'gaps': [], 'limitations': result['limitations'],
        'issues': [], 'provenance': [], 'progress': [], 'discovery': cast(EnzymeDiscovery, preview),
    }
    return outcome(result['status'], view, [], result['limitations'], 'COMPUTED')


def run_offline(
    question: str,
    *,
    on_update: Callable[[ToolUpdate], None] | None = None,
    csv_path: str = str(ROOT / "data/nrel/label_data.csv"),
    cohort_origin: str | None = None,
    cohort_filter: str | None = None,
    cohort_compare_filters: tuple[str, str] | None = None,
    cohort_compare_scopes: tuple[bool, bool] | None = None,
    cohort_pazy_only: bool = False,
    cohort_window: ExperimentalWindow | None = None,
    context_turns: list[str] | None = None,
) -> ToolResult[ResearchView]:
    """Run an example or supported PET-enzyme discovery question offline.

    Purpose: broad PET-enzyme discovery uses the independent pinned-source
        instrument. Other questions must match EXAMPLES exactly; unsupported
        free-form investigations remain explicit errors, not fabricated data.
    Args:
        question: an example or recognized broad PET-enzyme discovery question.
        on_update: optional progress observer.
        csv_path: trusted NREL CSV path for examples; discovery always uses pinned sources.
    Returns:
        ToolResult[ResearchView] via OfflineWorkflow.
    Example:
        >>> from petzyme.ui_adapter import EXAMPLES
        >>> out = run_offline(EXAMPLES['Temperature evidence'])
        >>> (out['status'], out['result']['workflow_status'])
        ('SUCCESS', 'COMPLETE')
    Flow: Receives external question str from Streamlit main(); returns ToolResult[ResearchView] from run_local_discovery() or investigate_with_backend(), without inference."""
    question = question.strip()
    try:
        interpret_request(question)
    except ValueError as error:
        return outcome('ERROR', None, [], [], None, str(error))
    try:
        backend = OfflineWorkflow(EXAMPLES["Temperature evidence"] if cohort_origin else question,
                                  context_turns=None if cohort_origin else context_turns)
    except ValueError as error:
        return outcome('ERROR', None, [], [], None, str(error))
    return investigate_with_backend(
        question, backend, on_update=on_update, csv_path=csv_path, cohort_origin=cohort_origin,
        automatic_runtime=True,
        cohort_filter=cohort_filter,
        cohort_compare_filters=cohort_compare_filters,
        cohort_compare_scopes=cohort_compare_scopes,
        cohort_pazy_only=cohort_pazy_only,
        cohort_window=cohort_window,
        context_turns=context_turns,
    )



_CONTEXT_RUNNER_REQUIRED = "A context-aware runner is required for this follow-up."


def _research_notice(status: Literal["ERROR", "NO_DATA"], message: str) -> ToolResult[ResearchView]:
    """Flow: Receives status and safe message from dispatch_research_turn(); returns ToolResult[ResearchView] for the research interface."""
    return outcome(status, None, [], [], None, message)


def _replace_research_label(question: str, target: str, replacement: str) -> str:
    """Flow: Receives user question and verified replacement from dispatch_research_turn(); returns str for its amended research context."""
    return re.sub(r"(?<![\w])" + re.escape(target) + r"(?![\w])", lambda _: replacement, question, flags=re.IGNORECASE)

def _evidence_matches_subjects(view: ResearchView, subjects: tuple[str, ...]) -> bool:
    """Flow: Receives ResearchView and resolved subject names from dispatch_research_turn(); returns bool for its final presentation guard."""
    allowed = {subject.casefold() for subject in subjects}
    if not allowed:
        return True
    def matches(rows: object, field: str) -> bool:
        if not isinstance(rows, list):
            return False
        return all(isinstance(row, dict) and isinstance(row.get(field), str)
                   and row[field].casefold() in allowed for row in rows)
    if not matches(view.get("observations", []), "enzyme_identifier"):
        return False
    multi = view.get("multi_study")
    if isinstance(multi, dict):
        nrel, pazy = multi.get("nrel"), multi.get("pazy")
        if isinstance(nrel, dict) and not matches(nrel.get("observations", []), "enzyme_identifier"):
            return False
        if isinstance(pazy, dict) and not matches(pazy.get("records", []), "protein_name"):
            return False
        coverage = multi.get("experimental_coverage")
        if isinstance(coverage, dict) and coverage.get("enzyme") not in (None, ""):
            if not isinstance(coverage["enzyme"], str) or coverage["enzyme"].casefold() not in allowed:
                return False
    return True


def dispatch_research_turn(
    session: ResearchSession, text: str,
    runner: Callable[[str], ToolResult[ResearchView]],
    *, context_runner: Callable[[str, ResearchSession], ToolResult[ResearchView]] | None = None,
    allow_adaptive: bool = False,
) -> tuple[ResearchSession, ToolResult[ResearchView] | ToolResult[IdentityTriage], bool]:
    """Flow: Receives ResearchSession and researcher text from the research interface; returns ResearchSession, transient scientific or identity ToolResult and replay flag for that interface."""
    if credential_like(text):
        return session, _research_notice("ERROR", "Credential-like input was withheld; use the existing secure credential store."), False
    if question_contains_private_input(text):
        updated = append_research_turn(session, text)
        identity = triage_protein(sequence_input=text, limit=5)
        value = identity["result"]
        matches = {c["identifier"] for c in value["candidates"]
                   if c["source_name"] == "NREL" and "EXACT_SEQUENCE" in c["basis"]} if value else set()
        label = (next(iter(matches)) if value and value["identity_status"] == "IDENTIFIED_SEQUENCE"
                 and len(matches) == 1 and not value["omitted_candidates"] else None)
        if label and session.pending_label:
            questions = tuple(_replace_research_label(q, session.pending_label, label) for q in session.active_questions)
            updated = replace(updated, active_questions=questions, pending_label=None, pending_candidate=None)
        return replace(updated, sequence_label=label), identity, False
    current_turn = interpret_request(text)
    if (not session.active_questions and not current_turn["enzyme_identifiers"]
            and re.search(r"\b(?:those|these|them|previous(?:ly)?)\b", text, re.IGNORECASE)):
        return session, _research_notice(
            "NO_DATA", "Please identify the enzymes or results you mean; there is no previous research group in this session."
        ), False
    if (session.discovery_replayable and re.search(r"\bcompar(?:e|ing|ison)\b", text, re.IGNORECASE)
            and re.search(r"\b(?:previous|prior|those|these)\b", text, re.IGNORECASE)):
        has_current_filter = bool(current_turn['temperatures'] or current_turn['ph_values'] or current_turn['pet_forms'])
        group_questions = ((session.historical_question, text) if has_current_filter else
                           (session.previous_historical_question, session.historical_question))
        if not all(group_questions):
            return session, _research_notice("ERROR", "Identify two historical condition groups for a comparability assessment; no reranked group was substituted."), False
        if _explicit_source_only(text, _NREL_ONLY_PATTERNS):
            return session, _research_notice("NO_DATA", "An NREL-only cohort comparison needs separately verified cross-source identities."), False
        current_source = (True if _explicit_source_only(text, _PAZY_ONLY_PATTERNS) else
                          False if re.search(r"\b(?:both sources|both nrel and pazy|both pazy and nrel)\b", text, re.IGNORECASE)
                          else session.cohort_pazy_only)
        group_sources = ((session.historical_pazy_only, current_source) if has_current_filter else
                         (session.previous_historical_pazy_only, session.historical_pazy_only))
        if None in group_sources:
            return session, _research_notice("NO_DATA", "A historical group's source restriction is not recorded; retrieve it again before comparison."), False
        if _explicit_source_only(text, _PAZY_ONLY_PATTERNS) and not all(group_sources):
            return session, _research_notice("NO_DATA", "A PAZy-only comparison would change an earlier group's recorded source scope; retrieve both groups as PAZy-only first."), False
        if context_runner is None and runner is not run_offline:
            return session, _research_notice("ERROR", _CONTEXT_RUNNER_REQUIRED), False
        if session.cohort_fingerprint != runtime_identity()['code_fingerprint']:
            return session, _research_notice("ERROR", "The code changed since the cohort was selected; start a new discovery."), False
        filters = cast(tuple[str, str], group_questions)
        scopes = cast(tuple[bool, bool], group_sources)
        updated = replace(append_research_turn(session, text), cohort_origin=session.active_questions[0],
                          cohort_compare_filters=filters, cohort_compare_scopes=scopes,
                          cohort_pazy_only=all(scopes))
        result = (context_runner(text, updated) if context_runner else
                  run_offline(text, cohort_origin=updated.cohort_origin,
                              cohort_compare_filters=filters, cohort_compare_scopes=scopes,
                              cohort_pazy_only=updated.cohort_pazy_only))
        if result['result'] is None:
            return session, result, True
        result['result']['submitted_question'] = text
        return replace(updated, cohort_origin=None, cohort_compare_filters=None, cohort_compare_scopes=None,
                       cohort_pazy_only=session.cohort_pazy_only), result, True
    planning_turn = bool({"experiment_advice", "exploratory_planning"} & set(current_turn["objectives"]))
    if session.discovery_replayable and session.active_questions and planning_turn:
        if context_runner is None and runner is not run_offline:
            return session, _research_notice("ERROR", _CONTEXT_RUNNER_REQUIRED), False
        window = current_turn["window"]
        updated = replace(append_research_turn(session, text), cohort_origin=session.active_questions[0],
                          cohort_filter=session.historical_question, objectives=tuple(current_turn["objectives"]),
                          available_temperatures_c=tuple(window["temperatures_c"]) or session.available_temperatures_c,
                          maximum_ph=window["maximum_ph"] if window["maximum_ph"] is not None else session.maximum_ph)
        planning_window: ExperimentalWindow = {**window, "temperatures_c": list(updated.available_temperatures_c),
                                                "maximum_ph": updated.maximum_ph}
        updated = replace(updated, cohort_window=planning_window)
        result = (context_runner(text, updated) if context_runner else
                  run_offline(text, cohort_origin=updated.cohort_origin, cohort_filter=updated.cohort_filter,
                              cohort_pazy_only=updated.cohort_pazy_only, cohort_window=updated.cohort_window))
        if result["result"] is None:
            return session, result, True
        result["result"]["submitted_question"] = text
        return replace(updated, cohort_origin=None, cohort_filter=None, cohort_window=None), result, True
    if (session.discovery_replayable and re.search(r"\b(?:equipment|device|can only|only run)\b", text, re.IGNORECASE)
            and not re.search(r"\b(?:tested|measured|recorded|reported|historical)\b", text, re.IGNORECASE)):
        window = current_turn["window"]
        updated = replace(append_research_turn(session, text),
                          available_temperatures_c=tuple(window["temperatures_c"]) or session.available_temperatures_c,
                          maximum_ph=window["maximum_ph"] if window["maximum_ph"] is not None else session.maximum_ph)
        return updated, _research_notice("NO_DATA", "Laboratory limits updated for future experiments; historical evidence filters are unchanged."), False
    historical_filter = bool(current_turn["temperatures"] or current_turn["ph_values"] or current_turn["pet_forms"])
    source_change = (_explicit_source_only(text, _PAZY_ONLY_PATTERNS)
                     or _explicit_source_only(text, _NREL_ONLY_PATTERNS)
                     or bool(re.search(r"\b(?:both sources|both nrel and pazy|both pazy and nrel)\b", text, re.IGNORECASE)))
    referential = re.search(r"\b(?:those|these|them|previous|what about|which have)\b", text, re.IGNORECASE)
    if (session.discovery_replayable and historical_filter and
            {"experiment_advice", "exploratory_planning"} & set(session.objectives) and
            re.search(r"\bwhat about\b", text, re.IGNORECASE) and
            not re.search(r"\b(?:tested|measured|observed|recorded|historical|evidence)\b", text, re.IGNORECASE)):
        return session, _research_notice("NO_DATA", "Clarify whether these conditions concern historical evidence or a future experiment."), False
    if (session.discovery_replayable and historical_filter and not referential and not source_change
            and not current_turn["enzyme_identifiers"] and "enzyme_discovery" not in current_turn["objectives"]):
        return session, _research_notice("ERROR", "Clarify whether the historical query concerns the previously selected enzymes or a new group."), False
    if (session.discovery_replayable and session.active_questions and (historical_filter or source_change)
            and (referential or source_change)
            and not ({"experiment_advice", "exploratory_planning"} & set(current_turn["objectives"]))):
        if context_runner is None and runner is not run_offline:
            return session, _research_notice("ERROR", _CONTEXT_RUNNER_REQUIRED), False
        if session.cohort_fingerprint != runtime_identity()["code_fingerprint"]:
            return session, _research_notice("ERROR", "The code changed since the cohort was selected; start a new discovery."), False
        origin = session.active_questions[0]
        requested_pazy_only = (_explicit_source_only(text, _PAZY_ONLY_PATTERNS)
                               if source_change else session.cohort_pazy_only)
        updated = replace(append_research_turn(session, text), cohort_origin=origin,
                          cohort_filter=session.historical_question,
                          previous_historical_question=(session.historical_question if historical_filter else session.previous_historical_question),
                          previous_historical_pazy_only=(session.historical_pazy_only if historical_filter else session.previous_historical_pazy_only),
                          historical_question=text if historical_filter else session.historical_question,
                          historical_pazy_only=(requested_pazy_only if historical_filter else session.historical_pazy_only),
                          objectives=("multi_study",),
                          cohort_pazy_only=requested_pazy_only)
        result = (context_runner(text, updated) if context_runner else
                  run_offline(text, cohort_origin=origin, cohort_filter=updated.cohort_filter,
                              cohort_pazy_only=updated.cohort_pazy_only))
        if result["result"] is None:
            return session, result, True
        result["result"]["submitted_question"] = text
        return replace(updated, cohort_origin=None, cohort_filter=None), result, True
    if session.pending_label and re.fullmatch(r"\s*no(?:[,.!].*)?\s*", text, re.IGNORECASE):
        rejected = (*session.rejected_candidates, session.pending_candidate) if session.pending_candidate else session.rejected_candidates
        updated = replace(append_research_turn(session, text), pending_candidate=None, rejected_candidates=rejected)
        return updated, _research_notice("NO_DATA",
            "Please provide an accession, publication DOI, exact name or sequence for " + session.pending_label + ". No identity was substituted."), False
    if session.pending_label and session.pending_candidate and re.fullmatch(r"\s*yes[.!]?\s*", text, re.IGNORECASE):
        if context_runner is None and runner is not run_offline:
            return session, _research_notice("ERROR", _CONTEXT_RUNNER_REQUIRED), False
        confirmed = session.pending_candidate
        prior = tuple(_replace_research_label(q, session.pending_label, confirmed) for q in session.active_questions)
        updated = replace(append_research_turn(session, text), active_questions=prior,
                          pending_label=None, pending_candidate=None)
        result = context_runner(text, updated) if context_runner else run_offline(text, context_turns=list(prior))
        return (updated if result["result"] is not None else session), result, False
    if session.pending_label:
        labels = interpret_request(text)["enzyme_identifiers"]
        new_objective = bool(labels and re.search(r"\b(?:investigate|find|search|study)\b", text, re.IGNORECASE))
        if len(labels) == 1 and text.strip().casefold() == labels[0].casefold():
            supplied = labels[0]
            match = triage_protein(name=supplied, limit=5)["result"]
            if match and any(c["name"].casefold() == supplied.casefold() and "EXACT_NAME" in c["basis"]
                             for c in match["candidates"]):
                if context_runner is None and runner is not run_offline:
                    return session, _research_notice("ERROR", _CONTEXT_RUNNER_REQUIRED), False
                old_label = session.pending_label
                prior = tuple(_replace_research_label(q, old_label, supplied) for q in session.active_questions)
                updated = replace(append_research_turn(session, text), active_questions=prior,
                                  pending_label=None, pending_candidate=None)
                result = context_runner(text, updated) if context_runner else run_offline(text, context_turns=list(prior))
                return (updated if result["result"] is not None else session), result, False
        if not labels or ("instead" not in text.lower() and not new_objective):
            updated = append_research_turn(session, text)
            return updated, _research_notice("NO_DATA",
                "Identity remains unresolved. Please provide an accession, publication DOI, exact name or sequence for " + session.pending_label + "."), False
        session = replace(session, pending_label=None, pending_candidate=None, active_questions=())
    if session.pending_need == "FOCUS":
        label = text.strip()
        focus_matches = [item for item in session.enzyme_identifiers if item.casefold() == label.casefold()]
        new_labels = interpret_request(text)["enzyme_identifiers"]
        if not focus_matches and new_labels and re.search(r"\b(?:investigate|find|search|study)\b", text, re.IGNORECASE):
            session = replace(session, active_questions=(), pending_need=None, dialogue_context=None,
                              discovery_replayable=False)
        elif len(focus_matches) != 1:
            updated = append_research_turn(session, text)
            return updated, _research_notice("NO_DATA",
                "Please choose one of the named enzymes for the existing single-enzyme advisor; no winner is assumed."), False
        else:
            if context_runner is None and runner is not run_offline:
                return session, _research_notice("ERROR", _CONTEXT_RUNNER_REQUIRED), False
            updated = replace(append_research_turn(session, text), active_questions=(*session.active_questions, text.strip()), pending_need=None,
                              enzyme_identifiers=(focus_matches[0],), discovery_replayable=False, dialogue_context=None)
            result = context_runner(text, updated) if context_runner else run_offline(text, context_turns=list(session.active_questions))
            return (updated if result["result"] is not None else session), result, False
    sequence_followup = False
    if not session.active_questions and session.sequence_label and re.search(
        r"\b(?:this|that|its|sequence|protein|enzyme)\b", text, re.IGNORECASE
    ):
        session = replace(session, active_questions=("Investigate " + session.sequence_label + ".",))
        sequence_followup = True
    if (not session.active_questions and session.turns and session.turns[-1].kind == "private_input"
            and not session.sequence_label and not interpret_request(text)["enzyme_identifiers"]):
        return session, _research_notice("ERROR",
            "The sequence identity remains unresolved; provide a source-backed identifier before investigating its evidence."), False
    current_request = interpret_request(text)
    if (session.active_questions and current_request["enzyme_identifiers"]
            and not re.search(r"\b(?:those|these|them|that|this|their|previous|compare with|compare another|also|instead of)\b", text, re.IGNORECASE)):
        session = replace(session, active_questions=(), discovery_replayable=False,
                          pending_label=None, pending_candidate=None, sequence_label=None)
    # A planning turn without an explicitly named subject refers to the active
    # resolved investigation; the wording need not match one fixed phrase.
    subjectless_planning = bool({"experiment_advice", "exploratory_planning"} & set(current_request["objectives"])) and not current_request["enzyme_identifiers"]
    if subjectless_planning and not session.enzyme_identifiers and not session.discovery_replayable:
        return session, _research_notice("NO_DATA", "Please identify the enzyme before requesting an experimental plan."), False
    source_followup = sequence_followup or bool(session.active_questions and not current_request["enzyme_identifiers"]
                           and (subjectless_planning or re.search(r"\b(?:pazy|nrel|both sources)\b", text, re.IGNORECASE)))
    updated, question, followup = prepare_research_turn(session, text, source_followup=source_followup)
    if followup and context_runner is None and runner is not run_offline:
        return session, _research_notice("ERROR", _CONTEXT_RUNNER_REQUIRED), False
    try:
        parsed = contextualize_request(question, list(session.active_questions) if followup else [])
    except ValueError as error:
        return session, _research_notice("NO_DATA", str(error)), False
    sources = parsed["sources"]
    if (len(parsed["enzyme_identifiers"]) == 2
            and not _explicit_source_only(question, _NREL_ONLY_PATTERNS)
            and not _explicit_source_only(question, _PAZY_ONLY_PATTERNS)):
        sources = ["NREL", "PAZy"]
    equipment_turns = [q for q in updated.active_questions if re.search(
        r"\b(?:equipment|device|available temperatures|can only|only run|supports|can maintain|can operate)\b",
        q, re.IGNORECASE)]
    available = (interpret_request(equipment_turns[-1])["window"]["temperatures_c"]
                 if equipment_turns else [])
    if followup and not available:
        available = list(session.available_temperatures_c)
    maximum_ph = parsed["window"]["maximum_ph"]
    if (followup and current_request["window"]["maximum_ph"] is not None and
            re.search(r"\b(?:equipment|device|maximum\s+ph|cannot exceed|can only|only run)\b", text, re.IGNORECASE)):
        maximum_ph = current_request["window"]["maximum_ph"]
    if followup and maximum_ph is None:
        maximum_ph = session.maximum_ph
    updated = replace(updated, objectives=tuple(parsed["objectives"]),
                      enzyme_identifiers=tuple(parsed["enzyme_identifiers"]), sources=tuple(sources),
                      temperatures_c=tuple(parsed["window"]["temperatures_c"]),
                      available_temperatures_c=tuple(available),
                      ph_values=tuple(parsed["window"]["ph_values"]), pet_forms=tuple(parsed["window"]["pet_forms"]),
                      maximum_ph=maximum_ph)
    for label in interpret_request(text if followup else question)["enzyme_identifiers"]:
        name_result = triage_protein(name=label, limit=5)["result"]
        if name_result is None:
            return session, _research_notice("ERROR", "Identity sources are unavailable; no identity was inferred."), False
        if any(c["name"].casefold() == label.casefold() and
               ({"EXACT_NAME", "DOCUMENTED_ALIAS"} & set(c["basis"])) for c in name_result["candidates"]):
            continue
        candidates = {c["identifier"] for c in name_result["candidates"] if c["source_name"] == "NREL"
                      and c["identifier"] not in session.rejected_candidates}
        candidate = next(iter(candidates)) if len(candidates) == 1 and not name_result["omitted_candidates"] else None
        updated = replace(updated, pending_label=label, pending_candidate=candidate)
        message = ("Did you mean the NREL label " + candidate + " for " + label + "? Please answer yes or no."
                   if candidate else "Please provide an accession, publication DOI, exact name or sequence for " + label + ". No identity was substituted.")
        return updated, _research_notice("NO_DATA", message), False

    planning = bool({"experiment_advice", "exploratory_planning"} & set(updated.objectives))
    if (len(updated.enzyme_identifiers) == 2 and planning
            and (not updated.available_temperatures_c or updated.maximum_ph is None)):
        updated = replace(updated, pending_need="CONSTRAINTS")
        return updated, _research_notice("NO_DATA",
            "For a conditional two-enzyme experiment, please state the available temperatures and maximum pH. No winner or laboratory feasibility is assumed."), False
    if (len(updated.enzyme_identifiers) == 2 and planning
            and (not allow_adaptive or set(updated.sources) != {"NREL", "PAZy"})):
        updated = replace(updated, pending_need="FOCUS")
        return updated, _research_notice("NO_DATA",
            "Please choose one of the named enzymes for the existing single-enzyme advisor; no winner is assumed."), False
    updated = replace(updated, pending_need=None)

    previous_labels = interpret_request(session.active_questions[0])["enzyme_identifiers"] if followup else []
    if followup and not session.discovery_replayable and not previous_labels:
        return session, _research_notice("ERROR",
            "The previous result cannot be safely reconstructed. Please restate the enzyme or conditions."), False
    previous = run_offline(session.active_questions[0]) if followup and session.discovery_replayable else None
    if followup and session.discovery_replayable and (not previous or not previous["result"] or not previous["result"].get("discovery")):
        return session, _research_notice("ERROR",
            "The earlier source evidence could not be reconstructed; please restate the question."), True
    result = (context_runner(question, updated) if context_runner else
              run_offline(question, context_turns=list(session.active_questions)) if followup and not session.discovery_replayable else runner(question))
    view = result["result"]
    if view is None:
        return session, result, followup
    if not _evidence_matches_subjects(view, updated.enzyme_identifiers):
        return session, _research_notice("ERROR", "Retrieved evidence did not match the resolved enzyme; no results were substituted."), followup
    dialogue = view.get("dialogue")
    if dialogue and validate_dialogue_context(dialogue.get("context")):
        updated = replace(updated, dialogue_context=deepcopy(dialogue["context"]))
    if followup:
        if not session.discovery_replayable:
            view["submitted_question"] = text
            return updated, result, False
        prior_view = previous["result"] if previous else None
        if not prior_view or not same_discovery_referents(prior_view.get("discovery"), view.get("discovery")):
            return session, _research_notice("ERROR",
                "The earlier enzyme set could not be revalidated under the requested conditions; no substitute ranking is shown."), True
        view["submitted_question"] = text
        return replace(updated, discovery_replayable=True), result, True
    replayed = False
    if view.get("discovery"):
        baseline = run_offline(question)
        replayed = True
        baseline_view = baseline["result"]
        replayable = bool(baseline_view and same_discovery_referents(
            baseline_view.get("discovery"), view.get("discovery")))
        updated = replace(updated, discovery_replayable=replayable,
                          cohort_pazy_only=updated.sources == ("PAZy",),
                          cohort_fingerprint=runtime_identity()["code_fingerprint"] if replayable else None)
    return updated, result, replayed
