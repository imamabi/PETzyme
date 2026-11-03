"""Independent semantic validation with real local instruments, no inference/network."""

import copy

import pytest

from petzyme.advisor_candidates import CONTROLS
from petzyme.agent_evidence import project_evidence
from petzyme.agent_types import AgentState
from petzyme.decision import REQUIRED_CONTROLS, DecisionError, validate_experimental_decision
from petzyme.dialogue import prepare_dialogue_turn
from petzyme.observations import load_nrel, parse_nrel
from petzyme.results import outcome

QUESTION = (
    "Should I use ESM050 or LCC-ICCG? My equipment supports 40C and 60C, but I cannot exceed pH 7."
)


def session():
    return {
        "original_temperatures_c": [40, 60],
        "nrel_references": [],
        "omitted_reference_count": 0,
        "previous_alternatives": [],
        "previous_ranking": [],
        "previous_rationale": [],
        "pazy_context_seen": False,
    }


@pytest.fixture
def state():
    _, c, _ = prepare_dialogue_turn({"context": None, "text": QUESTION, "input_kind": "ANSWER"})
    c["confirmed_labels"] = ["ESM050", "LCC-ICCG"]
    c["decision"] = session()
    s = AgentState("data/nrel/label_data.csv", dialogue_context=c)
    obs = parse_nrel(load_nrel()["result"])["result"]["observations"]
    s.decision_observations = [
        o
        for o in obs
        if o["enzyme_identifier"] in c["confirmed_labels"]
        and o["temperature_c"] in (40, 60)
        and o["ph"] == 6.5
        and o["pet_form"] == "CRYSTALLINE_POWDER"
    ]
    _, s.facts = project_evidence(
        "real",
        "filter_observations",
        outcome("SUCCESS", s.decision_observations, [], [], "MEASURED"),
    )
    return s


def option(s, id="P1", design="BALANCED_ENZYME_TEMPERATURE"):
    varied = (
        ["control condition"]
        if design == "MEASUREMENT_QUALIFICATION"
        else ["enzyme identity"]
        + (["temperature"] if design == "BALANCED_ENZYME_TEMPERATURE" else [])
    )
    return {
        "option_id": id,
        "design": design,
        "enzymes": ["ESM050"] if design == "MEASUREMENT_QUALIFICATION" else ["ESM050", "LCC-ICCG"],
        "temperatures_c": [40, 60] if design == "BALANCED_ENZYME_TEMPERATURE" else [40],
        "ph": 6.5,
        "pet_form": "CRYSTALLINE_POWDER",
        "controlled_variables": [v for v in CONTROLS if v not in varied],
        "varied_variables": varied,
        "controls": sorted(REQUIRED_CONTROLS),
        "evidence_ids": [f["id"] for f in s.facts if f["kind"] == "EVIDENCE"],
    }


def selection(s):
    ids = [f["id"] for f in s.facts if f["kind"] in ("EVIDENCE", "PROPOSAL")]
    return {
        "action": "EXPERIMENTAL_DECISION",
        "alternatives": [option(s), option(s, "P2", "MATCHED_ENZYME_COMPARISON")],
        "ranking": ["P1", "P2"],
        "rationale": [
            {
                "preferred": "P1",
                "over": "P2",
                "criterion": "TEMPERATURE_COVERAGE",
                "evidence_ids": ids,
            }
        ],
        "fact_ids": ids,
    }


def validate(v, s):
    return validate_experimental_decision(
        v, s, [f["id"] for f in s.facts if f["kind"] in ("EVIDENCE", "PROPOSAL")]
    )


def test_model_choices_stay_hypothetical_with_original_values_and_unknown_metadata(state):
    v, result = validate(selection(state), state)
    assert v["grounded"] and not v["complete"]
    assert result["selected_id"] == "P1" and result["status"] == "PARTIAL"
    assert result["evidence_category"] == "HYPOTHETICAL"
    assert result["comparability"]["quantitative_comparison_supported"] is False
    assert "UNITS_UNVERIFIED" in {i["code"] for i in result["evidence_assessment"]["issues"]}
    assert all(
        a["metadata_status"]["duration"] == "UNKNOWN"
        and a["resource_feasibility"]["status"] == "UNKNOWN"
        for a in result["alternatives"]
    )
    cells = {
        o["temperature_c"]: o["original_text"]
        for o in state.decision_observations
        if o["enzyme_identifier"] == "ESM050"
    }
    assert cells == {40: "0.08499999999999999", 60: "0.13"}


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "controls",
        "factors",
        "units",
        "winner",
        "missing_ids",
        "ranking",
        "reason",
        "ph",
        "temperature",
        "duplicate",
    ],
)
def test_unsupported_or_infeasible_recommendations_are_rejected(state, mutation):
    s = selection(state)
    if mutation == "identity":
        s["alternatives"][0]["enzymes"] = ["ESM050", "LCCMC"]
    if mutation == "controls":
        s["alternatives"][0]["controls"] = []
    if mutation == "factors":
        s["alternatives"][0]["controlled_variables"].append("temperature")
    if mutation == "units":
        s["alternatives"][0]["measurement_units"] = "umol/mg"
    if mutation == "winner":
        s["winner"] = "ESM050"
    if mutation == "missing_ids":
        s["fact_ids"] = []
    if mutation == "ranking":
        s["ranking"] = ["P1"]
    if mutation == "reason":
        s["rationale"][0]["criterion"] = "INVENTED_CONFIDENCE"
    if mutation == "ph":
        s["alternatives"][0]["ph"] = 8
    if mutation == "temperature":
        state.dialogue_context["temperatures_c"] = [40]
    if mutation == "duplicate":
        s["alternatives"][1] = copy.deepcopy(s["alternatives"][0])
        s["alternatives"][1]["option_id"] = "P2"
    with pytest.raises(DecisionError):
        validate(s, state)


def test_changed_temperature_constraints_allow_model_to_choose_a_different_option(state):
    old = selection(state)
    state.dialogue_context["decision"]["previous_alternatives"] = copy.deepcopy(old["alternatives"])
    state.dialogue_context["temperatures_c"] = [40]
    s = selection(state)
    s["ranking"] = ["P2", "P1"]
    s["rationale"] = [
        {
            "preferred": "P2",
            "over": "P1",
            "criterion": "EQUIPMENT_FEASIBILITY",
            "evidence_ids": s["fact_ids"],
        }
    ]
    _, result = validate(s, state)
    assert result["selected_id"] == "P2"
    assert result["previous_option_status"]["P1"] == "BLOCKED_BY_UPDATED_TEMPERATURES"
    assert result["alternatives"][0]["equipment_status"] == "BLOCKED"


def test_measurement_qualification_can_be_a_model_selected_distinct_alternative(state):
    s = selection(state)
    s["alternatives"][0] = option(state, "P1", "MEASUREMENT_QUALIFICATION")
    s["rationale"][0]["criterion"] = "MEASUREMENT_READINESS"
    _, result = validate(s, state)
    assert result["selected_id"] == "P1"
    assert "measurement" in result["alternatives"][0]["candidate"]["objective"]


def test_failed_source_authentication_cannot_support_any_plan(state, monkeypatch):
    monkeypatch.setattr("petzyme.decision.load_authenticated_nrel_reference", lambda: None)
    with pytest.raises(DecisionError, match="NREL_UNAVAILABLE"):
        validate(selection(state), state)


def test_per_option_citations_cannot_be_only_other_enzyme_or_warning(state):
    s = selection(state)
    s["alternatives"][0]["evidence_ids"] = [
        f["id"] for f in state.facts if f["text"].startswith("ESM050:")
    ]
    with pytest.raises(DecisionError, match="CITE_RETRIEVED_MEASUREMENTS"):
        validate(s, state)


def test_retrieving_only_one_enzyme_does_not_establish_a_two_candidate_decision(state):
    state.decision_observations = [
        o for o in state.decision_observations if o["enzyme_identifier"] == "ESM050"
    ]
    with pytest.raises(DecisionError, match="BOTH_CANDIDATES"):
        validate(selection(state), state)


def test_bounded_retrieval_gap_does_not_claim_global_measurement_absence(state):
    s = selection(state)
    for p in s["alternatives"]:
        p["ph"] = 5.5
    _, result = validate(s, state)
    assert all(
        "GLOBAL_AVAILABILITY_UNKNOWN" in c
        for a in result["alternatives"]
        for cs in a["condition_coverage"].values()
        for c in cs
    )
    assert all(
        any("remaining source coverage" in p for p in a["candidate"]["prerequisites"])
        for a in result["alternatives"]
    )


def test_formatted_temperature_strings_remain_invalid_rather_than_silently_coerced(state):
    s = selection(state)
    s["alternatives"][0]["temperatures_c"] = ["40°C", "60°C"]
    with pytest.raises(DecisionError, match="TEMPERATURE_LEVELS_MUST_BE_NUMERIC"):
        validate(s, state)
