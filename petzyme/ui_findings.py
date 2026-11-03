"""Presentation only: readable source values and emphasis over existing scientific assessments."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TypedDict, cast

from .contracts import JSONValue, UnitRepresentation
from .ui_adapter import ResearchView


def _display_observation(observation: dict[str, JSONValue]) -> tuple[str, str]:
    """Flow: Receives allowlisted observation dict from evidence_snapshot() or finding_answer(); returns display response/unit tuple for those callers, retaining original values in provenance."""
    working = cast(UnitRepresentation | None, observation.get('working_units'))
    if working and working['status'] == 'INFERRED_SUPPORTED':
        return display_response(str(working['standardized_value'])), working['display_unit'] + ' (INFERRED conversion; COMPUTED)'
    return display_response(str(observation['original_text'])), observation['unit_status']

# Display bounds, not scientific sampling or activity-ranking thresholds.
SNAPSHOT_ROWS_PER_FORM = 3
VISIBLE_ACTIONS = 3


@dataclass(frozen=True)
class DisplayStep:
    text: str
    rationale: str
    label: str
    is_experiment: bool = False


class SnapshotRow(TypedDict):
    enzyme: str
    temperature: str
    ph: str
    response: str


class SimilarityRow(TypedDict):
    protein: str
    cosine: str


@dataclass(frozen=True)
class EvidenceGroup:
    name: str
    rows: list[SnapshotRow]
    total: int
    units: str


@dataclass(frozen=True)
class ResultOverview:
    label: str
    answer: str
    attribution: str
    blocking: bool
    background_only: bool
    steps: list[DisplayStep]
    starting_points: list[str]
    uncertainties: list[str]


def _condition_number(value: object) -> str:
    return (
        f"{value:g}"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else "Unavailable"
    )


def _plan_steps(value: object) -> list[DisplayStep]:
    """Flow: Receives public plan object from result_overview() or finding_answer(); returns validated display-only DisplayStep list for those callers, without manufacturing missing proposal fields."""
    if not isinstance(value, dict) or not isinstance(value.get("recommendations"), list):
        return []
    steps = []
    for action in value["recommendations"]:
        if not isinstance(action, dict):
            continue
        text, rationale, feasibility = (
            action.get("proposed_action"),
            action.get("expected_information"),
            action.get("feasibility"),
        )
        if (
            not isinstance(text, str)
            or not isinstance(rationale, str)
            or not isinstance(feasibility, dict)
        ):
            continue
        status = feasibility.get("status")
        if isinstance(status, str):
            if (action.get('category') == 'PROPOSED_EXPERIMENT'
                    and text.startswith('PAZy already documents nominal target conditions')
                    and isinstance(action.get('target_conditions'), list)):
                targets = '; '.join(
                    f"{_condition_number(c['temperature_c'])}°C, pH {_condition_number(c['ph'])}, "
                    + str(c['pet_form']).replace('_', ' ').lower()
                    for c in action['target_conditions'])
                text = ('If source verification leaves uncertainty unresolved, consider a matched '
                        'verification/replication experiment at ' + targets + '. '
                        'These nominal conditions are already documented; this is not a novel-condition proposal.')
            steps.append(
                DisplayStep(
                    text,
                    rationale,
                    "HYPOTHETICAL · " + status + " feasibility; prerequisites apply",
                    action.get('category') == 'PROPOSED_EXPERIMENT',
                )
            )
    return steps


def _visible_plan_steps(value: object) -> list[DisplayStep]:
    """Flow: Receives executed plan object from result_overview(); returns bounded DisplayStep list including an available conditional experiment and its prerequisites for render_result(), without changing scientific priority."""
    steps = _plan_steps(value)
    if not isinstance(value, dict):
        return steps[:VISIBLE_ACTIONS]
    experiments = [i for i, action in enumerate(value.get("recommendations", []))
                   if action.get("category") == "PROPOSED_EXPERIMENT"]
    if experiments and experiments[0] >= VISIBLE_ACTIONS:
        return steps[:VISIBLE_ACTIONS - 1] + [steps[experiments[0]]]
    return steps[:VISIBLE_ACTIONS]


def similarity_snapshot(view: ResearchView) -> list[SimilarityRow]:
    """Flow: Receives computed ResearchView from present_investigation(); returns bounded SimilarityRow list for render_result(), without recomputing or interpreting protein similarity."""
    similarity = view["similarity"]
    raw = similarity.get("neighbors") if similarity else None
    if not isinstance(raw, list):
        return []
    rows = []
    for neighbor in raw[:SNAPSHOT_ROWS_PER_FORM]:
        if not isinstance(neighbor, dict):
            continue
        name, cosine = neighbor.get("identifier"), neighbor.get("cosine_similarity")
        if (
            isinstance(name, str)
            and isinstance(cosine, (int, float))
            and not isinstance(cosine, bool)
        ):
            rows.append(SimilarityRow(protein=name, cosine=f"{cosine:.6f}"))
    return rows


def evidence_snapshot(view: ResearchView) -> list[EvidenceGroup]:
    """Flow: Receives public ResearchView from present_investigation(); returns source-order EvidenceGroup list for render_result(), without ranking activities, pooling PET forms or changing original values."""
    groups = []
    keys = dict.fromkeys((str(o['pet_form']), _display_observation(o)[1]) for o in view['observations'])
    for form, unit in keys:
        observations = [o for o in view['observations'] if o['pet_form'] == form and _display_observation(o)[1] == unit]
        rows = [
            SnapshotRow(
                enzyme=str(o["enzyme_identifier"]),
                temperature=_condition_number(o["temperature_c"]),
                ph=_condition_number(o["ph"]),
                response=_display_observation(o)[0],
            )
            for o in observations[:SNAPSHOT_ROWS_PER_FORM]
        ]
        groups.append(
            EvidenceGroup(
                form.replace("_", " ").title(),
                rows,
                len(observations),
                unit,
            )
        )
    return groups


def result_overview(view: ResearchView) -> ResultOverview:
    """Flow: Receives validated ResearchView from present_investigation(); returns ResultOverview for render_result(), preserving unsupported scope and the distinction between observations and existing hypothetical proposals."""
    scope = view.get("scientific_scope")
    blocking = bool(
        scope
        and any(
            code in scope["issues"]
            for code in (
                "SCIENTIFIC_SCOPE_MATERIAL_MISMATCH",
                "SCIENTIFIC_SCOPE_QUANTITY_MISMATCH",
                "SCIENTIFIC_SCOPE_AMBIGUOUS_MATERIAL",
            )
        )
    )
    background = bool(scope and scope["background_only"])
    planning = bool(scope and scope["request"]["purpose"] == "EXPERIMENT_PLANNING")
    incomplete = view.get("synthesis", {}).get("status") == "INCOMPLETE"
    steps: list[DisplayStep] = []
    points: list[str] = []
    live = view["mode"] == "LIVE_META"
    multi = view.get('multi_study')
    multi_observed = bool(multi and (multi['nrel']['observations'] or
                          any(c['measurements'] for c in multi['pazy']['records'])))
    application_cohort = any(update.get("actor") == "APPLICATION_COVERAGE" and
                             update["call_id"].startswith("cohort_") for update in view["progress"])
    attribution = (
        (("Application retrieved cohort evidence; Muse selected canonical facts · deterministic scientific findings"
          if view.get("synthesis", {}).get("status") == "VALIDATED_FACT_SELECTION" else
          "Application retrieved cohort evidence; Muse synthesis is incomplete")
         if application_cohort else "Real Muse tool orchestration · deterministic scientific findings")
        if live
        else "Offline demonstration · deterministic tools; no live Muse reasoning"
        if view["mode"] == "DETERMINISTIC_OFFLINE"
        else "Offline/mock execution · not live Muse acceptance"
    )
    if blocking and scope is not None:
        label = "UNSUPPORTED COMPARISON · PARTIAL"
        answer = " ".join(scope["explanations"][:1])
        steps = [
            DisplayStep(
                "Clarify the scientific target",
                "Use evidence for the requested polymer and measurement quantity.",
                "CLARIFICATION",
            ),
            DisplayStep(
                "Review PET observations as background only",
                "They cannot establish another polymer's degradation or kinetic constants.",
                "EVIDENCE REVIEW",
            ),
        ]
    elif incomplete:
        label = "INCOMPLETE SYNTHESIS · PARTIAL"
        answer = (
            "Muse did not complete a final narrative answer in this run."
            if live
            else "The investigation did not complete an accepted model explanation."
        ) + (
            " The measured evidence below is still available."
            if view["observations"] or multi_observed
            else " No measured response was returned."
        )
        steps = [
            DisplayStep(
                "Review the evidence returned",
                "Deterministic tool findings remain separate from an accepted Muse answer.",
                "EVIDENCE REVIEW",
            ),
            DisplayStep(
                "Decide whether to continue",
                "Check the execution settings in Details; another live submission requires explicit authorization.",
                "WORKFLOW ACTION",
            ),
        ]
        if view['plan']:
            steps = _visible_plan_steps(view['plan'])
            attribution += ' · Tool-generated conditional proposals; Muse final synthesis is incomplete.'
    elif planning:
        label = "EXPLORATORY PLANNING · PARTIAL"
        answer = "The observations can inform a starting experiment. Confirm your laboratory constraints before choosing conditions; no proven optimum is established."
        steps = [
            DisplayStep(
                "Confirm your laboratory window",
                "Specify PET form, available temperatures, pH limits and the response you want to measure.",
                "CLARIFICATION",
            ),
            DisplayStep(
                "Verify the measurement basis",
                "Resolve unit, replication and protocol gaps before interpreting a comparison.",
                "VERIFICATION",
            ),
        ]
        if view["plan"]:
            steps = _visible_plan_steps(view["plan"])
            attribution += " · Conditional tool-generated proposals; laboratory feasibility is unconfirmed."
        # Show existing observed settings, never newly invented or activity-ranked proposals.
        for o in view["observations"]:
            point = f"{o['enzyme_identifier']} · {_condition_number(o['temperature_c'])}°C · pH {_condition_number(o['ph'])} · {str(o['pet_form']).replace('_', ' ').lower()}"
            if point not in points:
                points.append(point)
            if len(points) == 2:
                break
    else:
        label = (
            "HYPOTHETICAL PLAN · PARTIAL"
            if view["plan"]
            else "COMPUTED SIMILARITY"
            if view["similarity"]
            else "NO EXACT MEASUREMENTS · PARTIAL"
            if not view["observations"] and not multi_observed
            else "RECORDED EVIDENCE · PARTIAL"
            if view["workflow_status"] != "COMPLETE"
            or view["comparison"]
            and not view["comparison"]["direct_comparison_supported"]
            else "RECORDED EVIDENCE"
        )
        answer = finding_answer(view)
        if "publishable structured evidence" in answer:
            answer = "No measured response was returned in this run. Review the available source information and missing evidence."
        steps = _visible_plan_steps(view["plan"])
        if view["similarity"] and not steps:
            steps = [
                DisplayStep(
                    "Review the bounded neighbor search",
                    "Inspect the returned neighbors and candidate-window coverage in Details.",
                    "COMPUTED EVIDENCE",
                ),
                DisplayStep(
                    "Keep function separate from similarity",
                    "Representation similarity alone does not establish PET activity or experimental construct identity.",
                    "VERIFICATION",
                ),
            ]
        if not steps:
            steps = [
                DisplayStep(
                    "Inspect the source evidence",
                    "Check the requested conditions and the coverage shown below.",
                    "EVIDENCE REVIEW",
                ),
                DisplayStep(
                    "Resolve the remaining evidence gaps",
                    "Review measurement definitions, uncertainty and comparability before drawing a performance conclusion.",
                    "VERIFICATION",
                ),
            ]
    uncertainties = []
    if multi_observed:
        uncertainties.extend([
            'NREL source unit declarations remain unverified. Supported conversions are explicitly INFERRED / COMPUTED; PAZy reported units remain separate.',
            'Cross-source identity and experiment linkage are unresolved; same-publication curation is not independent replication.',
            'Response definitions, protocol comparability and uncertainty require source-specific verification.',
        ])
    if view["similarity"]:
        uncertainties.append(
            "Similarity does not establish PET activity or functional equivalence; only a bounded candidate window was searched."
        )
    if view["observations"]:
        units = {_display_observation(o)[1] for o in view["observations"]}
        uncertainties.append(
            "Response units: " + "; ".join(sorted(units)) + "; exact originals remain in Details."
        )
    limitations = " ".join(view["limitations"]).casefold()
    if "uncertainty" in limitations or "replicate" in limitations:
        uncertainties.append(
            "Replication and uncertainty are not established for all returned measurements."
        )
    if view["comparison"] and not view["comparison"]["direct_comparison_supported"]:
        uncertainties.append(
            "A direct quantitative comparison is unsupported; no causal effect is established."
        )
    elif planning:
        uncertainties.append("Comparability and laboratory suitability still require verification.")
    if view["plan"]:
        uncertainties.append(
            "Proposals are hypothetical; verify resources and prerequisites before any measurement."
        )
    elif view["observations"]:
        uncertainties.append(
            "Cell-linked protocol and publication metadata remain subject to verification."
        )
    if background and view["observations"]:
        uncertainties.insert(0, "Available PET observations are background only for this question.")
    return ResultOverview(
        label, answer, attribution, blocking, background, steps, points, uncertainties[:4]
    )


def display_response(original: str) -> str:
    """Format a recorded response for display (6 significant digits, trimmed).

    Purpose: display-only shortening; source precision stays intact in
        original_text/provenance and all scientific logic uses that.
    Flow: Receives original_text str from present_investigation(); returns six-significant-digit str for finding_answer() and render_findings(), retaining source precision elsewhere."""
    try:
        number = Decimal(original)
    except (InvalidOperation, ValueError):
        return original
    text = format(number, ".6g")
    return text.rstrip("0").rstrip(".") if "." in text and "e" not in text.lower() else text


def finding_answer(view: ResearchView) -> str:
    """Compose the deterministic headline answer from the view's grounded data.

    Purpose: pure function over observations/comparison/similarity/plan —
        no model prose. States what is and is not supported, including
        the explicit "no winning enzyme" style caveats.
    Flow: Receives ResearchView from present_investigation(); returns grounded answer str for render_findings(), without model-authored claims."""
    actions = _plan_steps(view["plan"])
    if actions:
        answer = "Start with this evidence check: " + actions[0].text
        answer += f" The plan contains {len(actions)} ordered actions. These are hypothetical proposals; no experiment has been performed."
        return ("Partial investigation. " if view["workflow_status"] != "COMPLETE" else "") + answer
    if view["similarity"]:
        similarity = view["similarity"]
        neighbors = similarity.get("neighbors")
        coverage = similarity.get("coverage")
        count = len(neighbors) if isinstance(neighbors, list) else 0
        searched = (
            coverage.get("searched_candidates") if isinstance(coverage, dict) else "unavailable"
        )
        answer = (
            f"ESM-2 returned {count} representation neighbors for {similarity['query_identifier']} "
            f"from {searched} searched candidates. This is computed similarity, not evidence of PET activity."
        )
        return ("Partial investigation. " if view["workflow_status"] != "COMPLETE" else "") + answer
    if not view["observations"] or not view["comparison"]:
        return view["headline"]
    observations = view["observations"]
    if len(observations) != 2:
        return view["headline"]
    descriptions = [
        f"{o['enzyme_identifier']} at {_condition_number(o['temperature_c'])}°C, pH {_condition_number(o['ph'])}, "
        f"{str(o['pet_form']).replace('_', ' ').lower()}: {_display_observation(o)[0]} {_display_observation(o)[1]}"
        for o in observations
    ]
    answer = "NREL records " + "; ".join(descriptions) + "."
    if not view["comparison"]["direct_comparison_supported"]:
        answer += " These recorded responses do not establish that one condition gives better PET activity; a direct quantitative comparison is unsupported."
    else:
        answer += " The documented group supports a quantitative comparison; it does not establish causality."
    if view["workflow_status"] != "COMPLETE":
        answer = "Partial investigation. " + answer
    return answer


def prioritize_limitations(view: ResearchView) -> tuple[list[str], list[str]]:
    """Flow: Receives ResearchView from present_investigation(); returns tuple[list[str], list[str]] for the primary and expandable limitations in render_findings()."""
    limits = list(dict.fromkeys(view["limitations"]))
    if view["similarity"]:
        topics = ("not measured pet activity", "whole-dataset", "no catalytic")
    elif view["plan"]:
        topics = ("hypothetical", "feasibility", "direct quantitative")
    elif any(c["status"] == "MISSING" for c in view["conditions"]):
        topics = ("missing", "not predicted", "units")
    else:
        topics = ("unit declarations", "units are unknown", "direct quantitative comparison", "uncertainty")
    selected = []
    for topic in topics:
        match = next((s for s in limits if topic in s.lower() and s not in selected), None)
        if match:
            selected.append(match)
    selected.extend(s for s in limits if s not in selected)
    return selected[:3], selected[3:]
