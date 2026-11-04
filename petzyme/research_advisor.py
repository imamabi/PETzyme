"""Transient evidence-to-advice composition over validated scientific instruments."""
from __future__ import annotations

from copy import deepcopy
from typing import Literal, TypedDict, cast

from .advisor import prioritize_experiments, recommend_next_experiments
from .advisor_feasibility import assess_experiment_feasibility, valid_constraints
from .advisor_types import EvidenceBasis, ExperimentCandidate, ExperimentPlan, ResearcherConstraints
from .assessment_types import ConditionInvestigation, EvidenceGap
from .conditions import investigate_conditions
from .contracts import Condition, ExperimentalWindow, JSONValue, PlanningGoal, ToolResult
from .evidence_records import CombinedExperimentalPreview, experimental_records
from .experimental import valid_condition
from .experimental_window import valid_window, within_window
from .gaps import identify_evidence_gaps
from .observations import load_nrel, parse_nrel
from .pet_evidence import ROOT, investigate_pet_evidence
from .pet_presenter import preview_pet_evidence
from .pet_retrieval import load_activity_source, match_pet_assays
from .privacy import safe_public_text
from .results import finite_number, outcome


class ResearchAdvice(TypedDict):
    answer: str
    condition_result: ToolResult[ConditionInvestigation]
    evidence_results: list[ToolResult[dict[str, JSONValue]]]
    gaps: list[EvidenceGap]
    plan_result: ToolResult[ExperimentPlan]
    scientific_status: Literal['PROPOSED_ONLY']
    partial_reasons: list[str]
    persistence: Literal['TRANSIENT_PAZY_SELECTION']


def apply_source_coverage(
    plan: ToolResult[ExperimentPlan], evidence: list[ToolResult[dict[str, JSONValue]]]
) -> ToolResult[ExperimentPlan]:
    """Flow: Receives ExperimentPlan envelope from recommend_next_experiments() and executed exact PAZy/NREL envelopes from investigate_pet_evidence(); returns transient ToolResult[ExperimentPlan] for advise_next_experiment(), qualifying documented targets as verification or replication rather than novel measurements.

    Does not fill NREL blanks, authenticate identity, change feasibility, remove
    prerequisites or pool responses. Missing/unavailable PAZy context is not absence.
    The returned plan inherits its parent's transient policy and must not be saved.
    """
    result = deepcopy(plan)
    if result['result'] is None:
        return result
    for action in result['result']['recommendations']:
        if action['category'] != 'PROPOSED_EXPERIMENT':
            continue
        documented = []
        unavailable = False
        for target in action['target_conditions']:
            form = 'powder' if target['pet_form'] == 'CRYSTALLINE_POWDER' else 'film'
            matched = [item for item in evidence if item['result'] is not None
                       and all(item['result']['query'].get(k) == v for k, v in
                               (('temperature_c', target['temperature_c']), ('ph', target['ph']), ('form', form)))]
            if not matched:
                unavailable = True
                continue
            for item in matched:
                value = item['result']
                if not value['pazy']['total_matches']:
                    continue
                documented.append(target)
                # Keep source/assay binding without forwarding private host file paths.
                sources = [{**source, 'locator': source['url'] + card['locator'],
                            'doi': card['publication_doi']}
                           for source in item['provenance'] if source['name'] == 'PAZy activity snapshot'
                           for card in value['pazy']['records']]
                action['evidence_basis'].append({
                    'category': 'COMPUTED',
                    'statement': f"PAZy documents {value['pazy']['total_matches']} nominal-condition assay(s) "
                    f"at {target}; source identity and original-experiment linkage remain unverified. "
                    'A missing condensed NREL cell is not a novel experimental condition.',
                    'sources': sources, 'csv_sha256': None,
                })
                recorded = [m['record'] for c in value['pazy']['records'] for m in c['measurements']
                            if m['record'].get('unit') == 'µmol aromatic products/mg enzyme'
                            and m['record'].get('value') is not None]
                if recorded:
                    action['evidence_basis'].append({
                        'category': 'MEASURED', 'statement': 'PAZy already reports individual product endpoints at '
                        + str(target) + ': ' + '; '.join(str(r['value']) + ' ' + r['unit'] for r in recorded)
                        + '. These are source-reported measurements, not converted NREL values or independent replication.',
                        'sources': sources, 'csv_sha256': None})
                for source in sources:
                    if source not in result['provenance']:
                        result['provenance'].append(source)
        if documented:
            action['proposed_action'] = (
                f'PAZy already documents nominal target conditions {documented}. '
                'Verify the source-linked construct, response definition and protocol first. '
                'Repeat documented targets only if unresolved comparability or uncertainty requires '
                'a matched verification/replication experiment; do not present them as novel conditions. '
                + action['proposed_action']
            )
            action['expected_information'] = (
                'Whether verification or independent replication resolves the uncertainty in an '
                'already-documented condition with reported endpoints, rather than measuring it as unexplored; '
                'quantify uncertainty and verify the source-linked assay/construct. Other targets remain separate. No activity outcome is predicted.'
            )
            action['limitations'].append(
                'PAZy nominal condition coverage does not prove sequence, construct or assay equivalence. '
                'Shared publications do not add independent replication.')
        if unavailable:
            action['limitations'].append(
                'PAZy coverage was unavailable for a target; novelty is not established.')
    return result


def advise_next_experiment(enzyme_identifier: str, condition_a: Condition | None = None, condition_b: Condition | None = None,
                           max_recommendations: int = 10,
                           optional_constraints: ResearcherConstraints | None = None,
                           source_scope: Literal['BOTH', 'PAZY_ONLY'] = 'BOTH',
                           request_window: ExperimentalWindow | None = None,
                           planning_goal: PlanningGoal = 'UNSPECIFIED') -> ToolResult[ResearchAdvice]:
    """Advise the next experiment using NREL evidence plus bounded PAZy context.

    Purpose: single high-level composition for "what should I do next?"
        Omit BOTH conditions for open-ended planning: inspect combined source
        coverage and outcomes, ask for laboratory constraints, and return
        conditional verification/replication examples. No optimum is inferred.
        When BOTH exact conditions are supplied, runs the NREL condition investigation and builds the ranked
        HYPOTHETICAL plan from it, and checks PAZy coverage using at most
        one assay projection per distinct exact condition. Documented
        targets are qualified as verification/replication, not novelty.
        PAZy never replaces NREL evidence, fills protocol metadata, or
        counts as replication or verified identity.
    Args:
        enzyme_identifier: exact nonempty label, at most 120 chars.
        condition_a: first exact Condition, or omit with condition_b for open planning.
        condition_b: second exact Condition; never invent an unspecified pair.
        max_recommendations: 1..10 plan prefix length.
        optional_constraints: validated ResearcherConstraints or None.
        source_scope: BOTH sources by default, or PAZY_ONLY for an explicit restriction.
    Returns:
        ToolResult[ResearchAdvice] with 'answer', 'condition_result',
            'evidence_results', 'gaps', 'plan_result',
            'scientific_status' (always PROPOSED_ONLY),
            'partial_reasons', and 'persistence'
            (TRANSIENT_PAZY_SELECTION — never save this object).
    Statuses:
        SUCCESS when no partial reason applies (rare); PARTIAL with
            explicit reasons otherwise; ERROR on bad inputs or
            unavailable local sources.
    Limitations:
        Open coverage advice never claims laboratory suitability. Exact mode uses
        only two requested conditions; cross-source comparison stays blocked
            until identity, units, and protocols verify.
    Flow: Receives external enzyme_identifier str, optional Condition pair and ResearcherConstraints; returns transient ToolResult[ResearchAdvice] for project_evidence() and public_tool_update()."""
    limits = [
        'Recommendations are HYPOTHETICAL; scientific assessments are COMPUTED and source responses remain MEASURED.',
        'The plan starts from NREL evidence gaps and checks PAZy target coverage before describing new measurements; documented targets require a specific verification or replication rationale. Sources are never pooled or authenticated by name.',
        'Resolve cross-source identity, original-experiment linkage, units and protocol gaps before any cross-source comparison; same-publication curation is not independent replication.',
        'Only two exact conditions are investigated; no prediction, unit conversion, whole-literature coverage or laboratory feasibility is inferred.',
        'PAZy selections and this integrated result are transient; no history, exports, caches or saved records.',
    ]
    if planning_goal not in ('UNSPECIFIED', 'VERIFY_ZERO', 'VERIFY_POSITIVE', 'COMPARE', 'EXPLORE'):
        return outcome('ERROR', None, [], limits, 'HYPOTHETICAL', 'Invalid experimental objective.')
    if request_window is not None and not valid_window(request_window):
        return outcome('ERROR', None, [], limits, 'HYPOTHETICAL', 'Invalid experimental window; no condition was inferred.')
    if (isinstance(enzyme_identifier, str) and enzyme_identifier.strip() and len(enzyme_identifier) <= 120
            and condition_a is None and condition_b is None
            and type(max_recommendations) is int and 1 <= max_recommendations <= 10
            and valid_constraints(optional_constraints) and source_scope in ('BOTH', 'PAZY_ONLY')):
        return _coverage_advice(enzyme_identifier, max_recommendations, optional_constraints, source_scope, request_window, planning_goal)
    if source_scope != 'BOTH':
        return outcome('ERROR', None, [], limits, 'HYPOTHETICAL', 'PAZy-only advice requires an open landscape; exact NREL condition investigation is excluded.')
    if (not isinstance(enzyme_identifier, str) or not enzyme_identifier.strip() or len(enzyme_identifier) > 120
            or any(not isinstance(c, dict) or set(c) != Condition.__annotations__.keys() or not valid_condition(c)
                   for c in (condition_a, condition_b))
            or type(max_recommendations) is not int or not 1 <= max_recommendations <= 10
            or not valid_constraints(optional_constraints)):
        return outcome('ERROR', None, [], limits, 'HYPOTHETICAL', 'Supply an exact enzyme, two valid conditions, a bound of 1–10 and structured constraints.')
    loaded = load_nrel(str(ROOT / 'data/nrel/label_data.csv'))
    if loaded['result'] is None:
        return outcome('ERROR', None, loaded['provenance'], limits, 'HYPOTHETICAL', 'Local NREL source unavailable; no fallback or download.')
    # A logical source locator retains file/cell provenance without exposing the host path.
    table = {**loaded['result'], 'source': {**loaded['result']['source'],
             'locator': loaded['result']['source']['url']+'#file=label_data.csv'}}
    parsed = parse_nrel(table)
    if parsed['result'] is None:
        return outcome('ERROR', None, parsed['provenance'], limits, 'HYPOTHETICAL', 'NREL parsing failed; no substitute evidence.')
    investigation = investigate_conditions(parsed['result']['observations'], condition_a, condition_b,
                                          enzyme_identifier=enzyme_identifier)
    if investigation['result'] is None:
        return outcome('ERROR', None, investigation['provenance'], limits, 'HYPOTHETICAL', investigation['message'])
    plan = recommend_next_experiments(investigation['result'], max_recommendations, optional_constraints)
    if plan['result'] is None:
        return outcome('ERROR', None, plan['provenance'], limits, 'HYPOTHETICAL', plan['message'])
    evidence, seen = [], set()
    for condition in (condition_a, condition_b):
        key = (condition['temperature_c'], condition['ph'], condition['pet_form'])
        if key in seen: continue  # Repeated conditions are never independent replication.
        seen.add(key)
        query = {'protein_name': enzyme_identifier, 'temperature_c': condition['temperature_c'],
                 'ph': condition['ph'], 'form': 'powder' if condition['pet_form'] == 'CRYSTALLINE_POWDER' else 'film',
                 'morphology': 'crystalline' if condition['pet_form'] == 'CRYSTALLINE_POWDER' else 'amorphous'}
        evidence.append(investigate_pet_evidence(query, limit=1))
    plan = apply_source_coverage(plan, evidence)
    plan = apply_experimental_window(plan, request_window)
    gaps = identify_evidence_gaps(investigation['result'])['result']
    reasons = []
    if not investigation['result']['direct_comparison_supported']: reasons.append('NREL_COMPARABILITY_UNRESOLVED')
    if investigation['status'] != 'SUCCESS': reasons.append('NREL_REQUESTED_MEASUREMENTS_MISSING')
    if plan['result']['omitted_count']: reasons.append('RECOMMENDATIONS_TRUNCATED')
    for item in evidence:
        if item['result'] is None:
            reasons.append('PAZY_CONTEXT_UNAVAILABLE')
        else:
            reasons.extend(item['result']['partial_reasons'])
            if not item['result']['pazy']['returned']: reasons.append('NO_EXACT_PAZY_MATCH')
            if item['result']['identity']['status'] != 'EXACT_SEQUENCE_MATCH': reasons.append('CROSS_SOURCE_IDENTITY_UNVERIFIED')
    sources = []
    for item in [investigation, plan, *evidence]:
        limits.extend(item['limitations'])
        for source in item['provenance']:
            if source not in sources: sources.append(source)
    observed = sum(bool(c['observations']) for c in investigation['result']['conditions'])
    answer = (f'Original NREL responses are available for {observed} of the two requested condition entries. '
              + ('The deterministic assessment supports a documented descriptive comparison, without causal attribution. '
                 if investigation['result']['direct_comparison_supported'] else
                 'They do not establish a meaningful activity difference or temperature superiority. ')
              + 'Verify source-linked measurement definitions, units, protocol and uncertainty first; '
              'if those records cannot resolve the comparison, consider the conditional matched-assay proposal. '
              'PAZy target coverage qualifies proposed repeats; a missing NREL cell alone does not establish novelty. '
              'PAZy curation is separate context, not independent replication or verified protein identity.')
    value: ResearchAdvice = {'answer': answer, 'condition_result': investigation, 'evidence_results': evidence,
        'gaps': gaps, 'plan_result': plan, 'scientific_status': 'PROPOSED_ONLY',
        'partial_reasons': list(dict.fromkeys(reasons)), 'persistence': 'TRANSIENT_PAZY_SELECTION'}
    return outcome('PARTIAL' if reasons else 'SUCCESS', value, sources, list(dict.fromkeys(limits)), 'HYPOTHETICAL')


def apply_experimental_window(plan: ToolResult[ExperimentPlan], window: ExperimentalWindow | None) -> ToolResult[ExperimentPlan]:
    """Flow: Receives conditional plan from apply_source_coverage() and explicit ExperimentalWindow from advise_next_experiment(); returns ToolResult[ExperimentPlan] with out-of-window proposals marked BLOCKED, retaining all source evidence and unknown resources."""
    value = deepcopy(plan)
    if value['result'] is not None and window is not None:
        for action in value['result']['recommendations']:
            if any(not within_window(c,window) for c in action['target_conditions']):
                action['feasibility']['status'] = 'BLOCKED'
                action['feasibility']['blockers'].append('Target condition exceeds an explicit researcher temperature/pH/PET-form restriction.')
                action['feasibility']['basis'] += ' Explicit experimental-window restriction applies.'
    return value


def _coverage_advice(
    enzyme: str,
    limit: int,
    constraints: ResearcherConstraints | None,
    source_scope: Literal["BOTH", "PAZY_ONLY"],
    request_window: ExperimentalWindow | None,
    planning_goal: PlanningGoal,
) -> ToolResult[ResearchAdvice]:
    """Flow: Receives an external enzyme str and validated bounds from advise_next_experiment(); returns transient ToolResult[ResearchAdvice] using investigate_pet_evidence(), assess_experiment_feasibility() and prioritize_experiments(), without inventing requested conditions or outcomes."""
    query = {'protein_name':enzyme}
    if request_window:
        if len(request_window['temperatures_c']) == 1:
            query['temperature_c'] = request_window['temperatures_c'][0]
        if len(request_window['ph_values']) == 1:
            query['ph'] = request_window['ph_values'][0]
        if len(request_window['pet_forms']) == 1:
            query['form'] = 'powder' if request_window['pet_forms'][0]=='CRYSTALLINE_POWDER' else 'film'
            query['morphology'] = 'crystalline' if query['form']=='powder' else 'amorphous'
    evidence = investigate_pet_evidence(query, limit=3, source_scope=source_scope)
    value = evidence["result"]
    if value is None:
        return outcome(
            "ERROR",
            None,
            evidence["provenance"],
            evidence["limitations"],
            "HYPOTHETICAL",
            evidence["message"],
        )
    inventory = value["experimental_coverage"]
    sources = [{**s, "locator": s["url"] + "#source-reference"} for s in evidence["provenance"]]
    basis = [
        {
            "category": "COMPUTED",
            "statement": value["answer"],
            "sources": sources,
            "csv_sha256": None,
        }
    ]
    common = experimental_records(cast(CombinedExperimentalPreview, preview_pet_evidence(value,safe_public_text)))
    for record in common:
        matching = [s for s in sources if (s['name']=='NREL') == (record['source']=='NREL')]
        source = [{**s, 'doi':record['publication'] or s['doi'], 'locator':s['url']+'#source-reference'} for s in matching]
        prefix = 'NREL original response' if record['source']=='NREL' else 'PAZy publication-derived endpoint:'
        basis.append({'category':'MEASURED',
            'statement':f"{prefix} {record['original_response']} {record['original_unit']}; conditions {record['temperature_c']}°C, pH {record['ph']}, {record['pet_form']}; definition {record['measurement_definition']}. No cross-source pooling or independent replication implied.",
            'sources':source, 'csv_sha256':None})
        qualified = record['qualified_unit']
        if qualified and qualified['status']=='INFERRED_SUPPORTED':
            basis.append({'category':'COMPUTED', 'statement':f"NREL qualified display: {qualified['standardized_value']} {qualified['display_unit']}; original unit {qualified['source_unit']}. Inferred from source reconciliation; original CSV declaration remains unverified.", 'sources':source,'csv_sha256':None})
    limitations = [
        *evidence["limitations"],
        "Explicit experimental limits restrict candidate conditions; unconfirmed resources remain UNKNOWN. Documented examples are not an established optimum or a validated protocol.",
    ]
    candidates: list[ExperimentCandidate] = []

    def add(
        code: str, text: str, *, experiment: bool = False, targets: list[Condition] | None = None,
        target_basis: list[EvidenceBasis] | None = None,
    ) -> None:
        existing = sum(candidate["code"] == code for candidate in candidates)
        action_id = "coverage-" + code.lower() + ("-" + str(existing + 1) if existing else "")
        candidates.append(
            {
                "action_id": action_id,
                "code": code,
                "category": "PROPOSED_EXPERIMENT" if experiment else "EVIDENCE_VERIFICATION",
                "evidence_category": "HYPOTHETICAL",
                "objective": f"Plan a source-grounded next measurement for candidate {enzyme}; research objective: {planning_goal.lower().replace('_', ' ')}.",
                "evidence_basis": deepcopy(basis + (target_basis or [])),
                "proposed_action": text,
                "required_measurements": [
                    "Define the response and calibration; report independent replication and uncertainty."
                ]
                if experiment
                else [],
                "target_conditions": targets or [],
                "controlled_variables": [
                    "Authenticate enzyme construct and PET material.",
                    "Match loading, duration, buffer and detection/normalization.",
                    "Include substrate-only and assay-background controls.",
                ]
                if experiment
                else [],
                "varied_variables": [],
                "expected_information": (
                    "Whether a measurement can be obtained under a condition absent from the selected-source grid; no outcome is predicted."
                    if code == "EXPLORE_UNRECORDED_CONDITION" else
                    "Whether independent measurement clarifies a recorded zero; no detection limit or outcome is invented."
                    if code == "VERIFY_RECORDED_ZERO" else
                    "Whether independent measurement clarifies a recorded positive response; no outcome is predicted."
                    if code == "REPLICATE_POSITIVE_RESPONSE" else
                    "Source-linked identity, assay interpretation and an explicit laboratory window before choosing an experiment."
                ),
                "prerequisites": [
                    c["action_id"] for c in candidates if c["category"] == "EVIDENCE_VERIFICATION"
                ]
                if experiment
                else [],
                "required_resources": [
                    "ASSAY_PROTOCOL",
                    "ENZYME_MATERIAL",
                    "PET_MATERIAL",
                    "TEMPERATURE_CONTROL",
                    "PH_CONTROL",
                    "RESPONSE_MEASUREMENT",
                    "LAB_ACCESS",
                ]
                if experiment
                else ["SOURCE_DOCUMENTATION"],
                "information_value": "RESOLVES_BLOCKER",
                "limitations": limitations,
            }
        )

    add(
        "VERIFY_IDENTITY_AND_ASSAY",
        "Verify the source-listed construct, original publication/worksheet linkage, measurement definition and inferred versus reported units. Shared-study curation is not independent replication.",
    )
    add(
        "DEFINE_LABORATORY_WINDOW",
        "Confirm PET form, available temperatures, pH bounds, duration/endpoint and resources. "
        + ("Specify whether to verify a zero, replicate a positive response, compare conditions or explore an untested condition. "
           if planning_goal == 'UNSPECIFIED' else "Confirm source assay and laboratory feasibility before the requested measurement. ")
        + "No established optimum is identified.",
    )
    additional_evidence: list[ToolResult[dict[str, JSONValue]]] = []
    if planning_goal in ('VERIFY_ZERO', 'VERIFY_POSITIVE') and inventory:
        # Scan the already authorized local source without retaining records or
        # ranking values. A target must have an actual numeric, unqualified
        # source endpoint of the kind requested by the researcher.
        snapshot = load_activity_source()
        if snapshot['result'] is not None and inventory['pazy_candidate_records'] == 1:
            matches = match_pet_assays(snapshot['result'], {'protein_name': enzyme})
            selected: list[Condition] = []
            for ref in matches:
                assay = ref['assay']
                c, m = assay['conditions'], assay['material']
                t, ph = c.get('temperature_c'), c.get('ph')
                form = ('CRYSTALLINE_POWDER' if (m.get('form'), m.get('morphology')) == ('powder', 'crystalline')
                        else 'AMORPHOUS_FILM' if (m.get('form'), m.get('morphology')) == ('film', 'amorphous') else None)
                if not finite_number(t) or not finite_number(ph) or form is None:
                    continue
                target: Condition = {'temperature_c': float(t), 'ph': float(ph), 'pet_form': form}
                if not within_window(target, request_window) or target in selected:
                    continue
                measured = [m for m in assay['measurements'] if m.get('metric_type') == 'specific_activity'
                            and isinstance(m.get('unit'), str) and bool(m['unit'])
                            and m.get('qualifier') in (None, '', '=') and finite_number(m.get('value'))]
                if not any((m['value'] == 0 if planning_goal == 'VERIFY_ZERO' else m['value'] > 0) for m in measured):
                    continue
                selected.append(target)
                query = {'protein_name': enzyme, 'temperature_c': target['temperature_c'], 'ph': target['ph'],
                         'form': 'powder' if form == 'CRYSTALLINE_POWDER' else 'film',
                         'morphology': 'crystalline' if form == 'CRYSTALLINE_POWDER' else 'amorphous'}
                target_evidence = investigate_pet_evidence(query, limit=3, source_scope=source_scope)
                if target_evidence['result'] is None or not target_evidence['result']['pazy']['total_matches']:
                    continue
                target_records = experimental_records(cast(CombinedExperimentalPreview,
                    preview_pet_evidence(target_evidence['result'], safe_public_text)))
                target_basis: list[EvidenceBasis] = []
                for record in target_records:
                    if record['source'] != 'PAZy' or not record['measurement_definition'].startswith('specific_activity · '):
                        continue
                    try:
                        numeric = float(record['original_response'])
                    except ValueError:
                        continue
                    if not finite_number(numeric) or not (numeric == 0 if planning_goal == 'VERIFY_ZERO' else numeric > 0):
                        continue
                    locator = record['reference'].split(' · ', 1)[-1]
                    record_sources = [{**src, 'doi': record['publication'], 'locator': src['url'] + locator}
                                      for src in target_evidence['provenance'] if src['name'] == 'PAZy activity snapshot']
                    target_basis.append({'category': 'MEASURED',
                        'statement': f"PAZy reports {record['original_response']} {record['original_unit']} for {record['measurement_definition']} at {target}; source labels do not authenticate constructs.",
                        'sources': record_sources, 'csv_sha256': None})
                if not target_basis:
                    continue
                additional_evidence.append(target_evidence)
                target_text = f"{target['temperature_c']:g}°C, pH {target['ph']:g}, {target['pet_form'].replace('_', ' ').lower()}"
                goal_text = ('verifying a reported zero response' if planning_goal == 'VERIFY_ZERO' else
                             'independently replicating a documented positive response')
                caveat = ('The detection limit has not been verified here, and zero does not prove inactivity. '
                          if planning_goal == 'VERIFY_ZERO' else '')
                add('VERIFY_RECORDED_ZERO' if planning_goal == 'VERIFY_ZERO' else 'REPLICATE_POSITIVE_RESPONSE',
                    f"After confirming the source assay and laboratory feasibility, consider {goal_text} at {target_text}. "
                    "The source already documents this condition; verify construct, protocol and detection first. "
                    + caveat + "No result, optimum or causal effect is predicted.",
                    experiment=True, targets=[target], target_basis=target_basis)
                if len(selected) >= 2:
                    break
    elif planning_goal == 'EXPLORE' and inventory:
        missing = [c for c in inventory['missing_in_both'] if c['temperature_c'] is not None and c['ph'] is not None
                   and (c['form'], c['morphology']) in (('powder', 'crystalline'), ('film', 'amorphous'))]
        for c in missing:
            target = {'temperature_c': c['temperature_c'], 'ph': c['ph'],
                      'pet_form': 'CRYSTALLINE_POWDER' if c['form'] == 'powder' else 'AMORPHOUS_FILM'}
            if within_window(target, request_window):
                add('EXPLORE_UNRECORDED_CONDITION',
                    'Conditional proposal: investigate a condition absent from the selected-source observed grid. '
                    'This is not proof of novelty across the literature; confirm the assay and feasibility first.',
                    experiment=True, targets=[target])
                break
    elif planning_goal == 'COMPARE':
        add('VERIFY_COMPARABILITY',
            'Verify source-linked constructs, assay definitions and all controlled variables before selecting conditions for comparison. '
            'Do not pool responses or interpret a difference causally without documented comparability.')
    if planning_goal in ('VERIFY_ZERO', 'VERIFY_POSITIVE', 'EXPLORE') and not any(
        c['category'] == 'PROPOSED_EXPERIMENT' for c in candidates
    ):
        add('CLARIFY_UNAVAILABLE_TARGET',
            'No suitable target was verified in the bounded selected-source evidence and laboratory window. '
            'Clarify the objective or obtain source-linked evidence; no substitute result was invented.')
    assessments = [assess_experiment_feasibility(c, constraints) for c in candidates]
    if any(item["result"] is None for item in assessments):
        return outcome(
            "ERROR",
            None,
            sources,
            limitations,
            "HYPOTHETICAL",
            "Coverage proposal feasibility validation failed.",
        )
    ranked = prioritize_experiments(candidates, [item["result"] for item in assessments])
    if ranked["result"] is None:
        return outcome("ERROR", None, sources, limitations, "HYPOTHETICAL", ranked["message"])
    ordered = ranked["result"]
    plan: ExperimentPlan = {
        "recommendations": ordered[:limit],
        "candidate_count": len(ordered),
        "omitted_count": max(0, len(ordered) - limit),
        "omitted_categories": sorted({c["category"] for c in ordered[limit:]}),
        "basis_kind": "EXPERIMENTAL_COVERAGE",
        "scientific_status": "PROPOSED_ONLY",
        "limitations": limitations,
    }
    result: ResearchAdvice = {
        "answer": value["answer"]
        + " Next: confirm the laboratory window and source assay; any proposed repeat is conditional, not a novel condition or established optimum.",
        "condition_result": outcome(
            "NO_DATA", None, [], ["No exact condition pair was requested or invented."], "COMPUTED"
        ),
        "evidence_results": [evidence, *additional_evidence],
        "gaps": [{**g, "kind": "VERIFICATION", "target_conditions": []} for g in value["gaps"]],
        "plan_result": outcome("SUCCESS", plan, sources, limitations, "HYPOTHETICAL"),
        "scientific_status": "PROPOSED_ONLY",
        "partial_reasons": ["LABORATORY_FEASIBILITY_UNCONFIRMED", *value["partial_reasons"]],
        "persistence": "TRANSIENT_PAZY_SELECTION",
    }
    return outcome("PARTIAL", result, sources, limitations, "HYPOTHETICAL")
