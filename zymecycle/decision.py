"""Validate Muse's experimental choices; never generate or rank options for it.

The finite design vocabulary is an experimental grammar, not a proposal list.
Muse supplies alternatives, levels and preferences. Existing instruments retain
measurement interpretation, controls, gaps and resource-feasibility semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import cast, get_args

from .advisor_candidates import ADVISOR_LIMITS, ASSAY_RESOURCES, CONTROLS, MEASUREMENTS
from .advisor_feasibility import assess_experiment_feasibility
from .advisor_types import EvidenceBasis, ExperimentCandidate, Resource
from .agent_evidence import model_evidence
from .agent_types import AgentState, Validation
from .assessment_types import EvidenceGap
from .comparability import check_comparability
from .conditions import investigate_conditions
from .contracts import Condition, Observation
from .decision_schema import DecisionSchemaError, parse_decision_selection
from .decision_types import (
    AssessedOption,
    Control,
    Criterion,
    DecisionSchemaCheck,
    DecisionSelection,
    Design,
    ExperimentalDecision,
    ExperimentOption,
    PublicFact,
)
from .evidence import assess_evidence
from .experimental import unique_observations, valid_condition
from .gaps import identify_evidence_gaps
from .observations import parse_nrel
from .protein_triage import load_authenticated_nrel_reference

MAX_RETAINED_CELLS = 12  # Public NREL locators only; never a dataset/PAZy result cache.
REQUIRED_CONTROLS = set(get_args(Control))
DESIGNS = set(get_args(Design))
CRITERIA = set(get_args(Criterion))
LIMITS = [
    "HYPOTHETICAL conditional designs, not performed experiments or predicted outcomes.",
    "Source labels do not authenticate experimental constructs or PET activity.",
    "UNKNOWN units, loadings, duration, replication and uncertainty must be resolved, not inferred absent.",
    "Equipment compatibility does not establish laboratory/resource feasibility; no costs or success probabilities are invented.",
    "No quantitative enzyme winner, unit conversion, kinetic inference or causal effect is established.",
    "Repeated NREL cells are deduplicated for assessment; PAZy curation or repeated retrieval is not independent replication.",
    "Muse chooses designs and qualitative preferences; Python checks their meaning and renders canonical descriptions, not free scientific prose.",
]


class DecisionError(ValueError):
    """Safe static rejection category, never an echo of model/source content."""

    def __init__(self, code: str, schema_check: DecisionSchemaCheck | None = None):
        self.schema_check = schema_check
        super().__init__(code)


def retrieved_observations(state: AgentState) -> list[Observation]:
    """Flow: Receives executed AgentState from validate_experimental_decision(); returns authenticated list[Observation] for evidence instruments and retain_decision()."""
    loaded = load_authenticated_nrel_reference()
    if loaded is None:
        raise DecisionError("NREL_UNAVAILABLE")
    parsed = parse_nrel(loaded)["result"]
    if parsed is None:
        raise DecisionError("NREL_PARSE_FAILED")
    index = {
        (o["csv_sha256"], o["source_row"], o["source_column"]): o for o in parsed["observations"]
    }
    selected: list[Mapping[str, object]] = list(state.decision_observations)
    for event in state.events:
        result = state.results[event["call_id"]]["result"]
        if result is None or event["status"] == "ERROR":
            continue
        if event["name"] == "investigate_pet_evidence":
            selected.extend(result["nrel"]["observations"])
        elif event["name"] == "advise_next_experiment":
            selected.extend(o for item in result['evidence_results'] if item['result'] for o in item['result']['nrel']['observations'])
    observations = []
    for card in selected:
        sha, row, column = card["csv_sha256"], card["source_row"], card["source_column"]
        if (
            not isinstance(sha, str)
            or not isinstance(row, int)
            or isinstance(row, bool)
            or not isinstance(column, str)
        ):
            raise DecisionError("INVALID_SOURCE_CELL_REFERENCE")
        key = (sha, row, column)
        actual = index.get(key)
        actual_values: Mapping[str, object] = actual or {}
        if actual is None or any(
            card[k] != actual_values[k]
            for k in ("enzyme_identifier", "temperature_c", "ph", "pet_form", "original_text")
        ):
            raise DecisionError("SOURCE_CELL_MISMATCH")
        observations.append(actual)
    return unique_observations(observations)


def _parse(value: object) -> DecisionSelection:
    try:
        return parse_decision_selection(value)
    except DecisionSchemaError as error:
        first = error.check["issues"][0]
        # Preserve established temperature error codes while adding field/index/type diagnostics.
        code = error.code
        if first["field"] == "temperatures_c":
            if first["code"] == "WRONG_TYPE":
                code = (
                    "TEMPERATURE_LEVELS_MUST_BE_NUMERIC"
                    if first["expected"] == "number"
                    else "TEMPERATURE_ARRAY_REQUIRED"
                )
            elif first["code"] == "INVALID_COUNT":
                code = "EMPTY_OR_EXCESSIVE_TEMPERATURE_LEVELS"
        raise DecisionError(code, error.check) from None


def _assess(
    option: ExperimentOption, state: AgentState, observations: list[Observation]
) -> AssessedOption:
    context = state.dialogue_context
    assert context is not None
    names, temps = option["enzymes"], option["temperatures_c"]
    if len(names) != len(set(names)) or len(temps) != len(set(temps)):
        raise DecisionError("DUPLICATE_DESIGN_LEVEL")
    if (
        not names
        or not set(names) <= set(context["confirmed_labels"])
        or any(n in group for n in names for group in context["rejected_labels"])
    ):
        raise DecisionError("UNRESOLVED_OR_REJECTED_ENZYME")
    if any(not any(o["enzyme_identifier"] == n for o in observations) for n in names):
        raise DecisionError("EXECUTED_EVIDENCE_REQUIRED_FOR_EACH_TARGET")
    design = option["design"]
    if design == "MEASUREMENT_QUALIFICATION":
        if len(names) != 1 or len(temps) != 1:
            raise DecisionError("QUALIFICATION_REQUIRES_ONE_REFERENCE_CONDITION")
        varied = ["control condition"]
    else:
        if set(names) != set(context["confirmed_labels"]) or len(names) != 2:
            raise DecisionError("MATCHED_COMPARISON_REQUIRES_BOTH_IDENTITIES")
        if len(temps) != (2 if design == "BALANCED_ENZYME_TEMPERATURE" else 1):
            raise DecisionError("UNCONTROLLED_EXPERIMENTAL_DESIGN")
        varied = ["enzyme identity"] + (["temperature"] if len(temps) == 2 else [])
    controlled = set(CONTROLS) - set(varied)
    if set(option["controlled_variables"]) != controlled or set(option["varied_variables"]) != set(
        varied
    ):
        raise DecisionError("CONTROLLED_AND_VARIED_FACTORS_INCONSISTENT")
    if set(option["controls"]) != REQUIRED_CONTROLS:
        raise DecisionError("MISSING_OR_INVENTED_CONTROLS")
    conditions: list[Condition] = [
        {"temperature_c": t, "ph": option["ph"], "pet_form": option["pet_form"]} for t in temps
    ]
    if any(not valid_condition(c) for c in conditions):
        raise DecisionError("INVALID_PROPOSED_CONDITION")
    blockers = []
    if any(t not in context["temperatures_c"] for t in temps):
        blockers.append("Proposed temperature unavailable under current researcher constraints.")
    if (
        option["ph"] > context["maximum_ph"]
        or context["assay_ph"] is not None
        and option["ph"] != context["assay_ph"]
    ):
        blockers.append("Proposed pH conflicts with the external bound or explicit assay setpoint.")
    basis: list[EvidenceBasis] = []
    for o in observations:
        if o["enzyme_identifier"] in names:
            basis.append(
                {
                    "category": "MEASURED",
                    "statement": f"Original response {o['original_text']}; units {o['unit_status']}.",
                    "sources": [o["source"]],
                    "csv_sha256": o["csv_sha256"],
                }
            )
    # Reuse gap/condition instruments on retrieved cells only; never fill an unqueried condition.
    gaps: list[EvidenceGap] = []
    coverage: dict[str, list[str]] = {}
    for name in names:
        inv = investigate_conditions(
            [o for o in observations if o["enzyme_identifier"] == name],
            conditions[0],
            conditions[-1],
            enzyme_identifier=name,
        )
        if inv["result"] is not None:
            coverage[name] = [
                f"{c['condition']}: "
                + (
                    "OBSERVED_IN_RETRIEVED_EVIDENCE"
                    if c["status"] == "OBSERVED"
                    else "NOT_RETRIEVED; GLOBAL_AVAILABILITY_UNKNOWN"
                )
                for c in inv["result"]["conditions"]
            ]
            # A bounded preview is not the full CSV. Do not turn a retrieval gap into
            # a claim that the source lacks the measurement or requires a new assay.
            gaps.extend(
                g
                for g in (identify_evidence_gaps(inv["result"])["result"] or [])
                if g["kind"] == "VERIFICATION"
            )
    qualification = design == "MEASUREMENT_QUALIFICATION"
    objective = (
        "Qualify the response measurement before interpreting enzyme differences."
        if qualification
        else "Distinguish enzyme-associated response differences under a matched prospective design."
    )
    candidate: ExperimentCandidate = {
        "action_id": option["option_id"],
        "code": design,
        "category": "PROPOSED_EXPERIMENT",
        "evidence_category": "HYPOTHETICAL",
        "objective": objective,
        "evidence_basis": basis,
        "proposed_action": f"Conditional {design}: {names}; temperatures {temps}; proposed pH {option['ph']}; {option['pet_form']}.",
        "required_measurements": MEASUREMENTS.copy(),
        "target_conditions": conditions,
        "controlled_variables": option["controlled_variables"],
        "varied_variables": varied,
        "expected_information": "Whether blank/reference/calibration and uncertainty permit an interpretable response definition."
        if qualification
        else "Whether matched measurements distinguish the proposed enzyme/temperature hypotheses within estimated uncertainty.",
        "prerequisites": list(dict.fromkeys(g["requirement"] for g in gaps))
        + [
            "Check remaining source coverage before calling any not-retrieved condition a globally missing measurement."
        ],
        "required_resources": cast(list[Resource], ASSAY_RESOURCES.copy()),
        "information_value": "RESOLVES_BLOCKER" if qualification else "DIRECT_QUESTION",
        "limitations": ADVISOR_LIMITS.copy(),
    }
    assessed = assess_experiment_feasibility(candidate, state.researcher_constraints)["result"]
    if assessed is None:
        raise DecisionError("RESOURCE_FEASIBILITY_FAILED")
    return {
        "proposal": deepcopy(option),
        "candidate": candidate,
        "equipment_status": "BLOCKED" if blockers else "WITHIN_DECLARED_BOUNDS",
        "equipment_blockers": blockers,
        "resource_feasibility": assessed,
        "identity_status": {n: "SOURCE_LABEL_ONLY; CONSTRUCT_UNKNOWN" for n in names},
        "condition_coverage": coverage,
        "proposed_ph_origin": "OBSERVED_IN_RETRIEVED_SOURCE"
        if any(
            o["ph"] == option["ph"] and o["pet_form"] == option["pet_form"] for o in observations
        )
        else "HYPOTHETICAL_WITHIN_BOUND; NOT_OBSERVED",
        "metadata_status": {
            k: "UNKNOWN"
            for k in (
                "duration",
                "enzyme_loading",
                "substrate_loading",
                "replicate_count",
                "measurement_units",
            )
        },
        "hypothesis": "HYPOTHETICAL: the proposed endpoint can be distinguished from blank/reference uncertainty."
        if qualification
        else "HYPOTHETICAL: response may depend on enzyme identity"
        + (
            " and temperature, including their interaction."
            if len(temps) == 2
            else " at the proposed fixed temperature."
        ),
        "outcome_implications": [
            "HYPOTHETICAL: interpretable measurements could enable the subsequent matched comparison."
            if qualification
            else "HYPOTHETICAL: a difference resolved relative to uncertainty could support a condition-specific response difference, not general activity superiority.",
            "HYPOTHETICAL: a difference not resolved within uncertainty would not demonstrate equivalence.",
            "HYPOTHETICAL: failed blanks, controls or calibration would leave the decision unsupported.",
        ],
        "limitations": LIMITS.copy(),
    }


def validate_experimental_decision(
    value: object, state: AgentState, required_ids: list[str]
) -> tuple[Validation, ExperimentalDecision]:
    """Flow: Receives untrusted Muse selection and executed AgentState from run_investigation(); returns Validation and ExperimentalDecision for render_dialogue_result(), never a substitute model choice."""
    selection = _parse(value)
    context = state.dialogue_context
    if context is None or "decision" not in context:
        raise DecisionError("EXPLICIT_DECISION_MODE_REQUIRED")
    facts = {f["id"]: f for f in state.facts}
    ids = set(selection["fact_ids"])
    if not ids <= facts.keys() or not set(required_ids) <= ids:
        raise DecisionError("MISSING_OR_UNKNOWN_CANONICAL_IDS")
    observations = retrieved_observations(state)
    if any(n is None for n in context["confirmed_labels"]):
        raise DecisionError("RESOLVE_IDENTITIES_BEFORE_DECISION")
    if any(
        not any(o["enzyme_identifier"] == name for o in observations)
        for name in context["confirmed_labels"]
    ):
        raise DecisionError("EXECUTED_EVIDENCE_REQUIRED_FOR_BOTH_CANDIDATES")
    assessed = [_assess(o, state, observations) for o in selection["alternatives"]]
    signatures = {
        (
            a["proposal"]["design"],
            tuple(sorted(a["proposal"]["enzymes"])),
            tuple(sorted(a["proposal"]["temperatures_c"])),
            a["proposal"]["ph"],
            a["proposal"]["pet_form"],
        )
        for a in assessed
    }
    options = {a["proposal"]["option_id"]: a for a in assessed}
    if len(signatures) != len(assessed) or len(options) != len(assessed):
        raise DecisionError("SCIENTIFICALLY_DISTINCT_OPTIONS_REQUIRED")
    ranking = selection["ranking"]
    if len(ranking) != len(set(ranking)) or set(ranking) != set(options):
        raise DecisionError("RANK_ALL_OPTIONS_EXACTLY_ONCE")
    selected = options[ranking[0]]
    if selected["equipment_status"] == "BLOCKED" or selected["resource_feasibility"]["status"] in (
        "BLOCKED",
        "CONFLICTING_CONSTRAINTS",
    ):
        raise DecisionError("SELECTED_EXPERIMENT_INFEASIBLE")
    for option in assessed:
        refs = option["proposal"]["evidence_ids"]
        if not refs or not set(refs) <= ids:
            raise DecisionError("OPTION_EVIDENCE_REFERENCES_REQUIRED")
        for name in option["proposal"]["enzymes"]:
            if not any(
                facts[i]["kind"] == "EVIDENCE"
                and facts[i]["evidence_category"] == "MEASURED"
                and facts[i]["text"].startswith(name + ": recorded NREL response ")
                and any(s["name"] == "NREL" for s in facts[i]["provenance"])
                for i in refs
            ):
                raise DecisionError("CITE_RETRIEVED_MEASUREMENTS_FOR_EACH_TARGET")
    assessment = assess_evidence(observations)["result"]
    comparison = check_comparability(observations)["result"]
    if assessment is None or comparison is None:
        raise DecisionError("EVIDENCE_ASSESSMENT_FAILED")
    covered, explanations = set(), []
    for reason in selection["rationale"]:
        if (
            reason["preferred"] != ranking[0]
            or reason["over"] not in options
            or reason["over"] == ranking[0]
            or not reason["evidence_ids"]
            or not set(reason["evidence_ids"]) <= ids
        ):
            raise DecisionError("PREFERENCE_NOT_GROUNDED_IN_ALTERNATIVES")
        other = options[reason["over"]]
        p, q = selected["proposal"], other["proposal"]
        code = reason["criterion"]
        valid = {
            "EQUIPMENT_FEASIBILITY": other["equipment_status"] == "BLOCKED",
            "TEMPERATURE_COVERAGE": len(p["enzymes"]) == 2
            and len(p["temperatures_c"]) > len(q["temperatures_c"]),
            "FOCUSED_DESIGN": len(p["enzymes"]) * len(p["temperatures_c"])
            < len(q["enzymes"]) * len(q["temperatures_c"]),
            "DIRECT_ENZYME_DECISION": len(p["enzymes"]) == 2 and len(q["enzymes"]) < 2,
            "MEASUREMENT_READINESS": p["design"] == "MEASUREMENT_QUALIFICATION"
            and q["design"] != p["design"]
            and bool(
                {"UNITS_UNVERIFIED", "UNCERTAINTY_UNREPORTED", "REPLICATES_UNREPORTED"}
                & {i["code"] for i in assessment["issues"]}
            ),
        }[code]
        if not valid:
            raise DecisionError("PREFERENCE_CRITERION_NOT_SUPPORTED")
        covered.add(reason["over"])
        explanations.append(
            f"Muse preferred {ranking[0]} over {reason['over']} using {code}; Python verified the declared design/evidence relationship. No numerical information-gain or cost score was assigned."
        )
    if covered != set(options) - {ranking[0]}:
        raise DecisionError("EXPLAIN_EACH_REJECTED_ALTERNATIVE")
    previous = {
        p["option_id"]: (
            "BLOCKED_BY_UPDATED_TEMPERATURES"
            if not set(p["temperatures_c"]) <= set(context["temperatures_c"])
            else "STILL_WITHIN_TEMPERATURE_BOUNDS"
        )
        for p in context["decision"]["previous_alternatives"]
    }
    sources = []
    for o in observations:
        if o["source"] not in sources:
            sources.append(o["source"])
    result: ExperimentalDecision = {
        "status": "PARTIAL",
        "evidence_category": "HYPOTHETICAL",
        "alternatives": assessed,
        "selected_id": ranking[0],
        "feasible_option_ids": [
            i
            for i in ranking
            if options[i]["equipment_status"] != "BLOCKED"
            and options[i]["resource_feasibility"]["status"]
            not in ("BLOCKED", "CONFLICTING_CONSTRAINTS")
        ],
        "excluded_ids": [
            i
            for i in ranking
            if options[i]["equipment_status"] == "BLOCKED"
            or options[i]["resource_feasibility"]["status"]
            in ("BLOCKED", "CONFLICTING_CONSTRAINTS")
        ],
        "ranking": ranking,
        "rationale": selection["rationale"],
        "rationale_explanations": explanations,
        "evidence_assessment": assessment,
        "comparability": comparison,
        "previous_option_status": previous,
        "retained_reference_count": min(MAX_RETAINED_CELLS, len(observations)),
        "source_provenance": sources,
        "evidence_facts": cast(
            list[PublicFact],
            model_evidence(
                {"status": "SUCCESS", "refs": {}, "preview": {}},
                [
                    f
                    for f in state.facts
                    if f["id"] in ids
                    and f["evidence_category"] == "MEASURED"
                    and f["provenance"]
                    and all(s["name"] == "NREL" for s in f["provenance"])
                ],
            )["facts"],
        ),
        "limitations": LIMITS.copy(),
    }
    warnings = [f["id"] for f in state.facts if f["kind"] in ("LIMITATION", "UNANSWERED")]
    return {
        "grounded": True,
        "complete": False,
        "issues": [
            "Conditional experimental decision remains scientifically PARTIAL; no enzyme winner is established."
        ],
        "selected_fact_ids": list(dict.fromkeys(selection["fact_ids"] + warnings)),
        "selection_counts": {
            "required": len(required_ids),
            "selected": len(required_ids),
            "omitted": 0,
        },
    }, result


# This describes the output grammar and safeguards, not concrete proposals or a tool sequence.
DECISION_POLICY = (
    """
Adaptive experimental decision support is explicitly enabled for this dialogue.
You choose the actual scientific tools, experimental alternatives, levels, and qualitative priority.
Investigate both target source labels and use actual NREL/PAZy evidence before deciding.
Batch independent calls when useful within the existing budgets. Existing source instruments assess
units, comparability, gaps, PAZy duplication/identity and protocol limits; do not redo their calculations.
Here source-query filters and conditional per-enzyme advice are exploratory evidence requests, not
user-established assay setpoints. A maximum pH is not an exact pH. Retain UNKNOWN protocol/units.
Never pool sources or infer construct identity, replicated evidence, activity superiority or a causal effect.
Ask a validated clarification if identity or a necessary decision detail cannot be established.
Known source labels can support conditional planning even while constructs/units/protocols need verification;
keep those gaps as explicit prerequisites. Do not automatically stall on missing exact assay settings:
you may propose a HYPOTHETICAL pH/PET form within equipment bounds, distinct from a user-established setpoint.
Evidence IDs alone or a legacy explanation object do not complete this planning objective.
Otherwise finish with this structured outcome (no extra winner, confidence, cost, prose or numeric outcome):
{"action":"EXPERIMENTAL_DECISION","alternatives":[...],"ranking":["P1","P2"],
 "rationale":[{"preferred":"P1","over":"P2","criterion":"...","evidence_ids":[...]}],"fact_ids":[...]}
Use two or three scientifically distinct alternatives. Each option has exactly these fields:
option_id (P1..P9), design, enzymes (source-label array), temperatures_c (one/two numbers), ph (proposed,
within the user bound, HYPOTHETICAL until supplied/performed), pet_form (CRYSTALLINE_POWDER or AMORPHOUS_FILM),
controlled_variables, varied_variables, controls, evidence_ids. No measurement values, loadings, duration,
replicate counts or unverified units may be invented. Unknown quantities remain unknown, to verify/design.
Design grammar (you select/configure the options; Python does not generate or rank them):
MATCHED_ENZYME_COMPARISON: both confirmed labels, one temperature; vary enzyme identity.
BALANCED_ENZYME_TEMPERATURE: both labels crossed with both temperatures; vary enzyme identity and temperature.
MEASUREMENT_QUALIFICATION: one source label, one temperature; vary control condition to qualify response
measurement, blank/reference/calibration/detection and uncertainty before interpreting enzyme differences.
Every option holds all other controlled_variables constant. controlled_variables must be the exact control
vocabulary below minus varied_variables; qualifications keep all those variables controlled. For the
qualification varied_variables is ["control condition"]. For other designs use ["enzyme identity"] or
["enzyme identity","temperature"]. The chosen pH/PET form applies uniformly within each option, never
mix powder and film. Cite actual retrieved MEASURED NREL fact IDs for each target in that option.
controls must include all of SUBSTRATE_BLANK, REFERENCE_CONTROL, INDEPENDENT_REPLICATION,
UNCERTAINTY, DETECTION_LIMIT, MEASUREMENT_CALIBRATION. These are requirements, not invented completed work.
Select every required canonical source/proposal ID returned this turn, including restored NREL facts;
automatically retained warnings do not replace evidence. ranking includes every option exactly once.
For each lower-ranked option explain why the first option was preferred via at least one grounded criterion:
EQUIPMENT_FEASIBILITY (other option is blocked by current equipment), TEMPERATURE_COVERAGE (chosen
2-enzyme design covers more temperatures), FOCUSED_DESIGN (fewer proposed enzyme/temperature cells,
not a claim of lower cost), DIRECT_ENZYME_DECISION (chosen option contrasts both enzymes while the other
does not), MEASUREMENT_READINESS (chosen qualification addresses actual unit/uncertainty/replicate gaps).
Only select a current equipment-compatible, non-blocked option. Laboratory resources/costs remain UNKNOWN.
Design/hypothesis/outcome implications are HYPOTHETICAL and rendered canonically from these validated
choices. Your design and preference choices are not replaced by a deterministic recommendation.
When constraints change, preserve actual revalidated NREL facts and the prior hypothetical designs;
reassess them, revise/select alternatives as justified, and explain the change or continued preference.
Do not repeat already-resolved identity questions. Prior PAZy records are not retained or reusable proofs.
Controlled-variable vocabulary: """
    + repr(CONTROLS)
    + "\n"
)
