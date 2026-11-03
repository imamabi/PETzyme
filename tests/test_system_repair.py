"""Real pinned science, explicit offline model transport and adversarial contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from zymecycle.agent import run_investigation
from zymecycle.agent_request import interpret_request
from zymecycle.communication import canonical_explanation, validate_explanation
from zymecycle.evidence_records import experimental_records, reported_response
from zymecycle.experimental_window import parse_experimental_window, within_window
from zymecycle.pet_evidence import investigate_pet_evidence
from zymecycle.pet_presenter import preview_pet_evidence
from zymecycle.privacy import safe_public_text
from zymecycle.research_advisor import advise_next_experiment
from zymecycle.research_planning import needs_semantic_plan, validate_research_plan
from zymecycle.ui_adapter import run_offline

QUESTIONS = [
    "To get a high activity for PET breakdown what are the most proficient experiments?",
    "What are the five enzymes with the best response for PET breakdown?",
    "What experimental conditions tend to produce high PET degradation responses?",
    "Which PET hydrolases should I investigate first, and under what reported conditions?",
]


class NativeScript:
    """Engineering fixture only; it is not evidence of autonomous Muse behavior."""

    mode = "MOCK_META"

    def __init__(self, actions, *, plan=None, bad=None):
        self.actions, self.plan, self.bad = list(actions), plan, bad
        self.calls = 0
        self.messages = []

    def complete(self, messages, tools):
        self.calls += 1
        self.messages.append(deepcopy(messages))
        if tools[0]["function"]["name"] == "plan_research":
            name, args, identifier = "plan_research", self.plan, "intent"
        elif self.actions:
            name, args = self.actions.pop(0)
            identifier = "c" + str(self.calls)
        else:
            replies = []
            for m in messages:
                content = m["content"] or ""
                if m["role"] == "tool":
                    replies.append(json.loads(content))
                elif m["role"] == "system" and content.startswith("Application coverage safeguard"):
                    replies.append(json.loads(content[content.index("{") :]))
            last = replies[-1]
            required = last.get("final_selection", {}).get("required_fact_ids", [])
            identifiers = required or [f["id"] for r in replies for f in r.get("facts", [])]
            if self.bad == "omit":
                identifiers = identifiers[:-1]
            explanation = {"render": "CANONICAL_SELECTED"}
            if self.bad == "claim":
                explanation["claim"] = "60C causes superior PET activity."
            return {
                "role": "assistant",
                "content": json.dumps({"fact_ids": identifiers, "explanation": explanation}),
            }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": identifier,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }


@pytest.mark.parametrize("question", QUESTIONS)
def test_broad_natural_language_is_not_an_exact_preset_gate(question):
    result = run_offline(question)
    assert result["result"] is not None
    view = result["result"]
    assert view["discovery"]["entries"] and len(view["discovery"]["entries"]) == 5
    assert view["mode"] == "DETERMINISTIC_LOCAL_DISCOVERY"
    assert view["model_requests"] == 0
    assert {r["source"]: r["status"] for r in view["source_coverage"]} == {
        "NREL": "FOUND",
        "PAZy": "FOUND",
    }
    assert not view["discovery"]["global_ranking_supported"]


def test_new_semantic_intent_protocol_uses_the_same_scientific_agent():
    q = "For a PET bench screen, point me toward a useful starting panel."
    original = interpret_request(q)
    assert needs_semantic_plan(original)
    backend = NativeScript(
        [("discover_pet_enzymes", {})],
        plan={"objectives": ["enzyme_discovery"], "enzyme_identifiers": []},
    )
    report = run_investigation(q, backend)["result"]
    assert report["interpretation"]["status"] == "MUSE_VALIDATED"
    assert report["interpretation"]["model_attempts"] == 1
    assert [e["name"] for e in report["trace"]] == ["discover_pet_enzymes"]
    assert report["communication"]["status"] == "VERIFIED_EXTRACTIVE"
    assert report["status"] == "PARTIAL"  # Conditional group, never global superiority.


@pytest.mark.parametrize(
    "bad",
    [
        {"objectives": ["predict_activity"], "enzyme_identifiers": []},
        {"objectives": ["enzyme_discovery"], "enzyme_identifiers": ["ESM050"]},
        {"objectives": ["enzyme_discovery"], "enzyme_identifiers": [], "unit": "invented"},
        {"objectives": "enzyme_discovery", "enzyme_identifiers": []},
    ],
)
def test_model_plan_cannot_author_identity_science_or_bad_contract(bad):
    with pytest.raises(ValueError):
        validate_research_plan(
            bad,
            interpret_request("For a PET bench screen, point me toward a useful starting panel."),
        )


def test_application_coverage_prevents_silent_nrel_only_completion():
    q = "Investigate ESM050. What conditions and outcomes are reported, and what should I test next?"
    backend = NativeScript(
        [
            ("load_nrel", {}),
            ("parse_nrel", {"table_ref": "c1:table"}),
            (
                "filter_observations",
                {"observations_ref": "c2:observations", "enzyme_identifier": "ESM050"},
            ),
            ("assess_evidence", {"observations_ref": "c3:observations"}),
        ]
    )
    report = run_investigation(q, backend)["result"]
    assert {r["source"]: r["status"] for r in report["source_coverage"]} == {
        "NREL": "FOUND",
        "PAZy": "FOUND",
    }
    automatic = [e for e in report["trace"] if e.get("actor") == "APPLICATION_COVERAGE"]
    assert len(automatic) == 1 and automatic[0]["name"] == "investigate_pet_evidence"
    assert report["validation"]["selection_counts"]["omitted"] == 0
    assert report["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert report["communication"]["status"] == "VERIFIED_EXTRACTIVE"


@pytest.mark.parametrize("bad", [None, "omit", "claim"])
def test_compact_explanation_does_not_select_ids_or_accept_claims(bad):
    q = "Compare NREL and PAZy evidence for ESM050."
    report = run_investigation(
        q,
        NativeScript(
            [("investigate_pet_evidence", {"query": {"protein_name": "ESM050"}})], bad=bad
        ),
    )["result"]
    if bad == "omit":
        assert report["validation"]["selection_counts"]["omitted"] == 1
        assert report["communication"]["status"] == "DETERMINISTIC_FALLBACK"
    elif bad == "claim":
        assert report["communication"]["status"] == "DETERMINISTIC_FALLBACK"
    else:
        assert report["communication"]["status"] == "VERIFIED_EXTRACTIVE"
        sections = report["communication"]["sections"]
        assert sum(e["field"] == "measurement_summary" for e in sections["evidence"]) == 3
        assert all(e["sources"] for s in sections.values() for e in s)
        chosen = set(report["validation"]["selected_fact_ids"])
        assert {e["fact_id"] for s in sections.values() for e in s} <= chosen
        candidate = canonical_explanation([f for f in report["facts"] if f["id"] in chosen])
        candidate["evidence"][0]["text"] = "Temperature proves higher activity."
        assert (
            validate_explanation(candidate, report)["reason_code"] == "UNVERIFIED_WORDING_OR_ROLE"
        )


@pytest.mark.parametrize(
    "q,code",
    [
        (
            "Compare ESM050 on polypropylene powder at 40C and 60C, pH 6.5.",
            "SCIENTIFIC_SCOPE_MATERIAL_MISMATCH",
        ),
        (
            "Compare kcat/KM of ESM050 on PET powder at 40C and 60C, pH 6.5.",
            "SCIENTIFIC_SCOPE_QUANTITY_MISMATCH",
        ),
    ],
)
def test_scope_explanation_does_not_complete_unsupported_science(q, code):
    a = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
    report = run_investigation(
        q,
        NativeScript(
            [
                ("load_nrel", {}),
                ("parse_nrel", {"table_ref": "c1:table"}),
                (
                    "investigate_conditions",
                    {
                        "observations_ref": "c2:observations",
                        "enzyme_identifier": "ESM050",
                        "condition_a": a,
                        "condition_b": {**a, "temperature_c": 60},
                    },
                ),
            ]
        ),
    )["result"]
    assert report["synthesis"]["status"] == "INCOMPLETE"
    assert code in report["validation"]["issues"]
    assert not report["validation"]["complete"]
    assert report["validation"]["scientific_scope"]["background_only"]
    assert report["communication"]["status"] == "VERIFIED_BACKGROUND_EXPLANATION"


def test_source_only_receipt_excludes_without_consulting_other_source(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Excluded PAZy source was accessed")

    monkeypatch.setattr("zymecycle.pet_evidence.load_activity_source", forbidden)
    v = run_offline("Investigate NREL-only evidence for ESM050.")["result"]
    assert v["source_coverage"][1]["status"] == "EXCLUDED"
    assert v["source_coverage"][1]["query_attempts"] == 0
    assert v["observations"]


@pytest.mark.parametrize(
    "q", ["maximum pH is 7; only 40C now", "pH <= 7 and 40C", "40C; pH 7 or below"]
)
def test_partial_limits_are_not_setpoints_and_restrict_real_proposals(q):
    window = parse_experimental_window(q)
    assert window["ph_values"] == [] and window["maximum_ph"] == 7
    advice = advise_next_experiment("ESM050", request_window=window)["result"]
    actions = advice["plan_result"]["result"]["recommendations"]
    assert all(within_window(c, window) for a in actions for c in a["target_conditions"])
    assert all(
        a["evidence_category"] == "HYPOTHETICAL" and a["feasibility"]["status"] == "UNKNOWN"
        for a in actions
    )


def test_contradictory_limits_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="conflict"):
        parse_experimental_window("pH 8; maximum pH 7")
    with pytest.raises(ValueError, match="Conflicting"):
        parse_experimental_window("maximum pH 7; maximum pH 6")


def test_exact_advice_retains_outside_evidence_but_blocks_proposals():
    a = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
    b = {**a, "temperature_c": 60}
    window = parse_experimental_window("only 40C; maximum pH 7")
    advice = advise_next_experiment("ESM050", a, b, request_window=window)["result"]
    assert all(c["observations"] for c in advice["condition_result"]["result"]["conditions"])
    for action in advice["plan_result"]["result"]["recommendations"]:
        if any(not within_window(c, window) for c in action["target_conditions"]):
            assert action["feasibility"]["status"] == "BLOCKED"


def test_common_records_preserve_each_endpoint_and_qualified_unit_separately():
    raw = investigate_pet_evidence({"protein_name": "ESM050"})["result"]
    preview = preview_pet_evidence(raw, safe_public_text)
    records = experimental_records(preview)
    assert len(records) == len(preview["nrel"]["observations"]) + sum(
        len(c["measurements"]) for c in preview["pazy"]["records"]
    )
    nrel = [r for r in records if r["source"] == "NREL"]
    pazy = [r for r in records if r["source"] == "PAZy"]
    assert [r["original_response"] for r in nrel] == [
        o["original_text"] for o in preview["nrel"]["observations"]
    ]
    assert all(
        r["original_unit"] == "UNKNOWN/UNVERIFIED" and r["duration_value"] is None for r in nrel
    )
    assert all(r["qualified_unit"]["evidence_category"] == "COMPUTED" for r in nrel)
    assert all(
        r["qualified_unit"] is None and r["experiment_identity"] == "SOURCE_REFERENCE_ONLY"
        for r in pazy
    )
    assert [r["original_response"] for r in pazy] == [
        reported_response(m["record"])
        for c in preview["pazy"]["records"]
        for m in c["measurements"]
    ]
    assert all("replicate_n" in r["replicate_information"] for r in pazy)


def test_qualifiers_and_categorical_results_are_not_forced_to_numeric_values():
    from zymecycle.ui_findings import display_response

    assert display_response("Not documented") == "Not documented"
    assert display_response("< 0.1") == "< 0.1"
    assert reported_response({"value": 0, "unit": "%", "qualifier": "<"}) == "< 0"
    assert (
        reported_response({"value": None, "categorical_result": "no detected product"})
        == "no detected product"
    )


@pytest.mark.parametrize(('record', 'expected'), [
    ({'value': 0}, '0'),
    ({'value': 0.0}, '0.0'),
    ({'value': None}, 'Not recorded'),
    ({}, 'Not recorded'),
    ({'value': None, 'reported_result': '0'}, 'Reported text: 0'),
    ({'value': None, 'categorical_result': 'Not measured'}, 'Not measured'),
    ({'value': True}, 'Unavailable (invalid recorded value)'),
    ({'value': '0'}, 'Unavailable (invalid recorded value)'),
    ({'value': float('nan')}, 'Unavailable (invalid recorded value)'),
])
def test_source_response_missing_and_invalid_are_never_coerced_to_zero(record, expected):
    assert reported_response(record) == expected


def test_esm050_film_zero_is_present_in_pinned_source():
    from zymecycle.pet_retrieval import load_activity_source, match_pet_assays

    snapshot = load_activity_source()['result']
    assays = match_pet_assays(snapshot, {'protein_name': 'ESM050', 'temperature_c': 40,
                                        'ph': 4.5, 'form': 'film'})
    assert len(assays) == 1
    values = [m['value'] for m in assays[0]['assay']['measurements']]
    assert len(values) == 4 and all(type(v) in (int, float) and v == 0 for v in values)


def test_root_dataset_pin_rejects_valid_but_modified_csv(monkeypatch):
    from zymecycle.observations import load_nrel

    root = Path(__file__).resolve().parents[1]
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: b",sequence\nSynthetic,ACDEF\n"
        if p == root / "data/nrel/label_data.csv"
        else original(p),
    )
    assert load_nrel(str(root / "data/nrel/label_data.csv"))["status"] == "ERROR"


def test_scientific_runtime_failure_preserves_already_executed_evidence(monkeypatch):
    from zymecycle import agent_execution
    from zymecycle.ui_adapter import investigate_with_backend

    original = agent_execution.invoke_tool

    def fail(name, args):
        if name == "assess_evidence":
            raise RuntimeError("PRIVATE_TOOL_FAILURE_SENTINEL")
        return original(name, args)

    monkeypatch.setattr(agent_execution, "invoke_tool", fail)
    backend = NativeScript(
        [
            ("load_nrel", {}),
            ("parse_nrel", {"table_ref": "c1:table"}),
            (
                "filter_observations",
                {"observations_ref": "c2:observations", "enzyme_identifier": "ESM050"},
            ),
            ("assess_evidence", {"observations_ref": "c3:observations"}),
        ]
    )
    view = investigate_with_backend("Investigate NREL-only evidence for ESM050.", backend)["result"]
    assert view["observations"] and view["workflow_status"] == "PARTIAL"
    assert "PRIVATE_TOOL_FAILURE_SENTINEL" not in json.dumps(view)


def test_model_budget_exhaustion_does_not_erase_the_matching_query():
    from zymecycle.runtime_config import RuntimeLimits
    from zymecycle.ui_adapter import investigate_with_backend

    backend = NativeScript(
        [
            ("load_nrel", {}),
            ("parse_nrel", {"table_ref": "c1:table"}),
            (
                "filter_observations",
                {"observations_ref": "c2:observations", "enzyme_identifier": "ESM050"},
            ),
        ]
    )
    view = investigate_with_backend(
        "Investigate NREL-only evidence for ESM050.",
        backend,
        runtime_limits=RuntimeLimits(turn_models=3),
    )["result"]
    assert view["observations"] and view["synthesis"]["reason_code"] == "MODEL_REQUEST_LIMIT"
    assert view["synthesis"]["status"] == "INCOMPLETE"


def test_nonpreset_exact_advice_uses_both_sources_without_filling_conditions():
    view = run_offline(
        "What should I test next for ESM050 comparing 40C and 60C, pH 6.5, crystalline PET powder? Use NREL and PAZy."
    )["result"]
    assert view["advice"] and len(view["conditions"]) == 2
    assert {r["source"]: r["status"] for r in view["source_coverage"]} == {
        "NREL": "FOUND",
        "PAZy": "FOUND",
    }
    assert view["plan"]["scientific_status"] == "PROPOSED_ONLY"


def test_native_nrel_union_is_not_lost_behind_a_smaller_source_preview():
    from zymecycle.ui_adapter import investigate_with_backend

    backend = NativeScript(
        [
            ("load_nrel", {}),
            ("parse_nrel", {"table_ref": "c1:table"}),
            (
                "filter_observations",
                {"observations_ref": "c2:observations", "enzyme_identifier": "ESM050"},
            ),
            ("assess_evidence", {"observations_ref": "c3:observations"}),
        ]
    )
    v = investigate_with_backend(
        "Investigate ESM050. What conditions and outcomes are reported, and what should I test next?",
        backend,
    )["result"]
    assert len(v["observations"]) == 10
    assert len(v["multi_study"]["nrel"]["observations"]) == 6
    assert v["multi_study"]["accumulation"]["unique_nrel_retrieved"] == 10
    assert next(r for r in v["source_coverage"] if r["source"] == "NREL")["measurements"] == 10


def test_retained_source_restriction_is_not_silently_ignored():
    from zymecycle.dialogue import prepare_dialogue_turn
    from zymecycle.scientific_scope import ScientificScopeError

    with pytest.raises(ScientificScopeError, match="source restriction"):
        prepare_dialogue_turn(
            {
                "context": None,
                "input_kind": "ANSWER",
                "text": "Should I use ESM050 or LCC-ICCG with NREL-only evidence at 40C and 60C, maximum pH 7?",
            }
        )


def test_semantic_plan_preserves_explicit_source_scope():
    q = "NREL-only: for a PET bench screen, point me toward a useful starting panel."
    p = validate_research_plan(
        {"objectives": ["enzyme_discovery"], "enzyme_identifiers": []}, interpret_request(q)
    )
    assert p["sources"] == ["NREL"]


def test_broad_nonpreset_planning_keeps_the_integrated_advisor():
    view = run_offline(
        "Investigate ESM050. What experimental conditions and outcomes are reported, and what should I test next?"
    )["result"]
    assert view["advice"] and view["plan"]["recommendations"]
    assert view["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert {r["source"]: r["status"] for r in view["source_coverage"]} == {
        "NREL": "FOUND",
        "PAZy": "FOUND",
    }


def test_open_planning_does_not_hide_an_executed_conditional_experiment():
    from zymecycle.ui_findings import result_overview

    view = run_offline(
        "Investigate ESM050. What experimental conditions and outcomes are reported, and what should I test next?"
    )["result"]
    overview = result_overview(view)
    actions = view["plan"]["recommendations"]
    experiment = next(a for a in actions if a["category"] == "PROPOSED_EXPERIMENT")
    assert any(step.text == experiment["proposed_action"] for step in overview.steps)
    assert all(step.label.startswith("HYPOTHETICAL") for step in overview.steps)
    assert any("UNKNOWN" in step.label for step in overview.steps)
    assert "no proven optimum" in overview.answer


def test_replication_request_uses_combined_coverage_and_actual_equipment_limits():
    view = run_offline(
        "What experiment should I conduct to replicate ESM050 PET responses, with maximum pH 7 and only 40C now?"
    )["result"]
    assert view["plan"] and view["advice"]
    assert {r["source"]: r["status"] for r in view["source_coverage"]} == {
        "NREL": "FOUND",
        "PAZy": "FOUND",
    }
    proposed = [
        a for a in view["plan"]["recommendations"] if a["category"] == "PROPOSED_EXPERIMENT"
    ]
    assert proposed and all("not an unexplored condition" in a["proposed_action"] for a in proposed)
    assert all(
        c["temperature_c"] == 40 and c["ph"] <= 7 for a in proposed for c in a["target_conditions"]
    )
    assert all(a["feasibility"]["status"] == "UNKNOWN" for a in proposed)


def test_semantic_stage_call_id_cannot_be_reused_as_a_scientific_call():
    class DuplicatePlanId(NativeScript):
        def complete(self, messages, tools):
            value = super().complete(messages, tools)
            if self.calls == 2:
                value["tool_calls"][0]["id"] = "intent"
            return value

    backend = DuplicatePlanId(
        [("discover_pet_enzymes", {})],
        plan={"objectives": ["enzyme_discovery"], "enzyme_identifiers": []},
    )
    report = run_investigation(
        "For a PET bench screen, point me toward a useful starting panel.", backend
    )["result"]
    assert not report["trace"] and report["synthesis"]["status"] == "INCOMPLETE"
    assert report["tool_attempts"] == 0


def test_semantic_plan_cannot_duplicate_one_identity_as_two_targets():
    with pytest.raises(ValueError, match="INVALID_RESEARCH_PLAN_VALUE"):
        validate_research_plan(
            {"objectives": ["multi_study"], "enzyme_identifiers": ["ESM050", "esm050"]},
            interpret_request("Consider ESM050 for PET bench work."),
        )


def test_oneshot_form_uses_fresh_consent_and_current_accounting_without_replay(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from zymecycle import ui_live

    calls = []

    def local_worker(question, **kwargs):
        assert kwargs["authorized"] is True
        calls.append(question)
        view = run_offline("Compare NREL and PAZy evidence for ESM050.")
        view["result"]["mode"] = "LIVE_META"
        view["result"]["model_requests"] = 2
        view["result"]["provider_telemetry"] = []
        view["result"]["provider_budget"] = {"limit_usd": 10.0, "blocked": False}
        return view

    monkeypatch.setattr(ui_live, "run_keychain_investigation", local_worker)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py"), default_timeout=30
    ).run()
    app.radio(key="mode").set_value("Live Meta agent").run()
    q = "Compare NREL and PAZy evidence for ESM050."
    app.text_area(key="question").set_value(q)
    app.checkbox(key="authorize_1").check()
    app.button(key="investigate").click().run()
    assert not app.exception and calls == [q]
    assert app.checkbox(key="authorize_2").value is False
    assert "authorize_1" not in app.session_state
    assert "pending_oneshot" not in app.session_state and "view" not in app.session_state
    assert any("2 model" in c.value for c in app.caption)
    app.run()
    assert calls == [q] and not app.exception
    app.checkbox(key="authorize_2").check()
    app.button(key="investigate").click().run()
    assert not app.exception and calls == [q, q]
    assert app.checkbox(key="authorize_3").value is False


def test_actual_dialogue_view_shows_public_turn_not_internal_request_context():
    from test_decision_integration import PlanningTransport, first_payload, run

    payload = first_payload()
    first = run(payload, PlanningTransport())["result"]
    assert first["submitted_question"] == payload["text"]
    assert "Researcher decision context" not in first["submitted_question"]
    follow = {
        "context": first["dialogue"]["context"],
        "text": "I can only run experiments at 40C now.",
        "input_kind": "ANSWER",
    }
    second = run(follow, PlanningTransport(continuation=True))["result"]
    assert second["submitted_question"] == follow["text"]
    receipts = {r["source"]: r for r in second["source_coverage"]}
    assert receipts["NREL"]["status"] == "FOUND" and receipts["NREL"]["measurements"] == 4
    assert receipts["PAZy"]["status"] == "PENDING" and "not retained" in receipts["PAZy"]["reason"]
    assert second["tool_attempts"] == 0 and second["selection_counts"]["omitted"] == 0


def test_sequence_supplied_as_answer_is_not_echoed_as_the_display_question():
    import httpx
    from test_build10_acceptance import AMBIGUOUS
    from test_dialogue import QUESTION, DialogueTransport, run_turn

    from zymecycle.meta import MetaModelClient
    from zymecycle.ui_adapter import investigate_with_backend

    context = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"},
        DialogueTransport(["ESM050", "LCCMC"]),
    )["result"]["dialogue"]["context"]
    scenario = DialogueTransport(sequence=True)
    out = investigate_with_backend(
        "Local private input",
        MetaModelClient(transport=httpx.MockTransport(scenario.respond), completion_tokens=10000),
        dialogue_input={
            "context": context,
            "text": "Here is the sequence: " + AMBIGUOUS,
            "input_kind": "ANSWER",
        },
    )["result"]
    assert AMBIGUOUS not in json.dumps(out)
    assert (
        out["submitted_question"]
        == "Local identity clarification input (private sequence or reference withheld)."
    )


def test_provider_diagnostics_retain_counts_even_without_new_tool_findings():
    from streamlit.testing.v1 import AppTest

    script = """
from test_decision_integration import PlanningTransport,first_payload,run
from zymecycle.ui_runtime import render_provider_usage
first=run(first_payload(),PlanningTransport())["result"]
second=run({"context":first["dialogue"]["context"],"text":"I can only run experiments at 40C now.","input_kind":"ANSWER"},PlanningTransport(continuation=True))["result"]
render_provider_usage(second)
"""
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    text = " ".join(t.value for t in app.text)
    assert "Canonical evidence counts · required 4 · selected 4 · omitted 0" in text
    assert "cached None" in text and "model match True" in text and "total 180" in text
    assert "allowance 10000" in text and "request bytes" in text and "reasoning 40" in text
    assert not app.json and "view" not in app.session_state


@pytest.mark.parametrize("mode,code", [("schema", "ARGUMENT_SCHEMA"), ("target", "DIALOGUE_GUARD")])
def test_safe_tool_failure_stage_is_available_to_model_and_ui(mode, code):
    from zymecycle.agent_evidence import model_evidence, public_tool_update
    from zymecycle.agent_execution import execute_tool_call
    from zymecycle.agent_types import AgentState
    from zymecycle.dialogue import prepare_dialogue_turn
    from zymecycle.dialogue_rules import dialogue_catalog

    q, context, _ = prepare_dialogue_turn(
        {
            "context": None,
            "text": "Should I use ESM050 or LCC-ICCG? My device supports 40C and 60C, maximum pH 7.",
            "input_kind": "ANSWER",
        }
    )
    state = AgentState("unused", dialogue_context=context)
    args = (
        {"enzyme": "ESM050"}
        if mode == "schema"
        else {"query": {"protein_name": "PRIVATE_FAILURE_TARGET_SENTINEL"}}
    )
    event = execute_tool_call(
        {
            "id": "bad",
            "type": "function",
            "function": {
                "name": "triage_protein" if mode == "schema" else "investigate_pet_evidence",
                "arguments": json.dumps(args),
            },
        },
        state,
        dialogue_catalog(interpret_request(q)),
    )
    assert event["status"] == "ERROR" and event["error_code"] == code
    reply = model_evidence(event, state.facts)
    update = public_tool_update(event, state.results["bad"], 1, 1)
    assert reply["error_code"] == update["error_code"] == code
    assert "did not complete" in update["summary"]
    assert "PRIVATE_FAILURE_TARGET_SENTINEL" not in json.dumps([reply, update])


def semantic_screen_view():
    from zymecycle.ui_adapter import investigate_with_backend

    q = "I’m arranging a PET bench screen. Which evidence-backed starting panel merits attention, with the assay settings that were actually used?"
    return investigate_with_backend(
        q,
        NativeScript(
            [("discover_pet_enzymes", {})],
            plan={"objectives": ["enzyme_discovery"], "enzyme_identifiers": []},
        ),
    )["result"]


def test_semantic_discovery_is_visible_when_only_the_metric_is_unspecified():
    from streamlit.testing.v1 import AppTest

    view = semantic_screen_view()
    assert view["scientific_scope"]["background_only"] and not view["scientific_scope"]["issues"]

    def render(v):
        from zymecycle.ui_results import render_result

        render_result(v, lambda: None)

    app = AppTest.from_function(render, args=(view,), default_timeout=30).run()
    assert not app.exception and len(app.table[0].value) == 5
    assert any("desired performance metric is unspecified" in c.value for c in app.caption)
    assert not app.json and not app.get("download_button")
    assert view["scientific_scope"]["background_only"]  # Display did not change scientific scope.


@pytest.mark.parametrize("name", ["Polypropylene", "Kinetics"])
def test_qualified_discovery_display_cannot_bypass_an_explicit_scope_mismatch(name):
    from streamlit.testing.v1 import AppTest
    from test_ui_clarity import case_view

    view = semantic_screen_view()
    blocked = case_view(name)
    view["scientific_scope"] = blocked["scientific_scope"]
    view["synthesis"] = blocked["synthesis"]

    def render(v):
        from zymecycle.ui_results import render_result

        render_result(v, lambda: None)

    app = AppTest.from_function(render, args=(view,), default_timeout=30).run()
    assert not app.exception and any("unsupported" in e.value for e in app.error)
    assert not app.table
    assert not any("CONDITIONAL WITHIN-GROUP RANKING" in c.value for c in app.caption)


def test_composed_discovery_does_not_hide_executed_advisor_or_combined_measurements():
    from streamlit.testing.v1 import AppTest

    from zymecycle.ui_adapter import investigate_with_backend

    # Controlled offline orchestration, never a claimed autonomous model acceptance.
    backend = NativeScript(
        [("discover_pet_enzymes", {}), ("advise_next_experiment", {"enzyme_identifier": "ESM050"})],
        plan={"objectives": ["enzyme_discovery", "experiment_advice"], "enzyme_identifiers": []},
    )
    # An explicitly unclassified question exercises the semantic-plan transport.
    q = "For a PET bench screen, point me toward a useful starting panel."
    view = investigate_with_backend(q, backend)["result"]
    assert view["plan"] and view["multi_study"] and view["discovery"]

    def render(v):
        from zymecycle.ui_results import render_result

        render_result(v, lambda: None)

    app = AppTest.from_function(render, args=(view,), default_timeout=30).run()
    assert not app.exception and len(app.table[0].value) == 5
    assert any("What were the results?" in m.value for m in app.markdown)
    assert any("Suggested next step" in m.value for m in app.markdown)
    assert any("UNKNOWN feasibility" in c.value for c in app.caption)


def test_discovery_and_planning_intents_are_not_overwritten_by_the_rank_shortcut():
    q = "Which five PET enzymes have the highest recorded responses, and what conditions should I test next?"
    request = interpret_request(q)
    assert {"enzyme_discovery", "experiment_advice", "exploratory_planning"} <= set(
        request["objectives"]
    )
    view = run_offline(q)["result"]
    assert view["discovery"]["entries"] and view["advice"] and view["plan"]["recommendations"]
    assert [u["name"] for u in view["progress"]] == [
        "discover_pet_enzymes",
        "advise_next_experiment",
    ]
    assert view["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert all(a["evidence_category"] == "HYPOTHETICAL" for a in view["plan"]["recommendations"])


def test_advisor_composition_cannot_claim_an_unexecuted_discovery_step():
    q = "For a PET bench screen, point me toward a useful starting panel."
    backend = NativeScript(
        [("advise_next_experiment", {"enzyme_identifier": "ESM050"})],
        plan={"objectives": ["enzyme_discovery", "experiment_advice"], "enzyme_identifiers": []},
    )
    report = run_investigation(q, backend)["result"]
    assert any(
        "Required evidence steps not completed: discover_pet_enzymes" == i
        for i in report["validation"]["issues"]
    )
    assert report["synthesis"]["status"] == "INCOMPLETE"
    assert report["synthesis"]["reason_code"] == "REQUIRED_INVESTIGATION_STEP_MISSING"
    assert report["communication"]["status"] == "DETERMINISTIC_FALLBACK"
    assert report["facts"] and all(r["status"] == "FOUND" for r in report["source_coverage"])


def test_representation_only_workflow_does_not_claim_missing_activity_retrieval():
    from zymecycle.agent_types import AgentState
    from zymecycle.source_coverage import source_receipts

    receipts = source_receipts(
        interpret_request("Find proteins similar to LCC-ICCG using ESM-2."), AgentState("unused")
    )
    assert receipts[0]["source"] == "NREL" and receipts[0]["status"] == "NOT_RELEVANT"
    assert receipts[0]["measurements"] == 0 and receipts[0]["query_attempts"] == 0
    assert "activity measurements" in receipts[0]["reason"]


def test_coarse_source_cue_does_not_hide_discovery_and_planning_capabilities():
    from zymecycle.agent_request import select_tool_catalog

    q = "I’m organizing a PET bench screen. Give me an evidence-backed enzyme shortlist, the documented assay settings, and a sensible follow-on bench study."
    request = interpret_request(q)
    assert request["objectives"] == [
        "multi_study"
    ]  # Conservative cue, not full semantic understanding.
    assert {t["function"]["name"] for t in select_tool_catalog(request)} == {
        "discover_pet_enzymes",
        "investigate_pet_evidence",
        "advise_next_experiment",
    }
    report = run_investigation(
        q,
        NativeScript(
            [
                ("discover_pet_enzymes", {}),
                ("advise_next_experiment", {"enzyme_identifier": "ESM050"}),
            ]
        ),
    )["result"]
    assert report["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert report["communication"]["status"] == "VERIFIED_EXTRACTIVE"
    assert report["validation"]["selection_counts"]["omitted"] == 0
    required = set(report["validation"]["selected_fact_ids"])
    assert all(
        f["id"] in required for f in report["facts"] if f["kind"] in ("EVIDENCE", "PROPOSAL")
    )


@pytest.mark.parametrize("bad", [None, "claim"])
def test_recovered_argument_error_does_not_invalidate_canonical_source_truth(bad):
    report = run_investigation(
        "Compare NREL and PAZy evidence for ESM050.",
        NativeScript(
            [
                ("investigate_pet_evidence", {"unexpected": True}),
                ("investigate_pet_evidence", {"query": {"protein_name": "ESM050"}}),
            ],
            bad=bad,
        ),
    )["result"]
    assert report["trace"][0]["error_code"] == "ARGUMENT_SCHEMA"
    assert report["trace"][1]["status"] == "PARTIAL"
    assert report["status"] == "PARTIAL"
    assert (
        "One or more tool calls failed or were blocked; no substitute evidence was generated."
        in report["validation"]["issues"]
    )
    assert report["validation"]["selection_counts"]["omitted"] == 0
    assert report["communication"]["status"] == (
        "DETERMINISTIC_FALLBACK" if bad else "VERIFIED_EXTRACTIVE"
    )


def test_unrecovered_failed_instrument_does_not_gain_verified_communication():
    report = run_investigation(
        "Compare NREL and PAZy evidence for ESM050.",
        NativeScript(
            [
                ("investigate_pet_evidence", {"unexpected": True}),
                ("advise_next_experiment", {"enzyme_identifier": "ESM050"}),
            ]
        ),
    )["result"]
    assert report["facts"] and report["validation"]["selection_counts"]["omitted"] == 0
    assert report["status"] == "PARTIAL"
    assert report["communication"]["status"] == "DETERMINISTIC_FALLBACK"


def test_whole_catalog_is_screened_without_relaxing_private_sequence_detection(monkeypatch):
    from zymecycle import agent
    from zymecycle.agent_request import select_tool_catalog

    original = select_tool_catalog

    def private_description(request):
        catalog = original(request)
        catalog[0]["function"]["description"] = ">synthetic-private\n" + "A" * 80
        return catalog

    monkeypatch.setattr(agent, "select_tool_catalog", private_description)
    backend = NativeScript([])
    report = run_investigation(
        "Compare ESM050 at 40C and 60C, pH 6.5, crystalline PET powder.", backend
    )["result"]
    assert backend.calls == 0 and not report["trace"]
    assert report["synthesis"]["status"] == "INCOMPLETE"
    assert "A" * 80 not in json.dumps(report)


def test_unnamed_semantic_experiment_plan_does_not_expose_live_protein_catalog():
    from zymecycle.agent_request import select_tool_catalog

    request = validate_research_plan(
        {"objectives": ["experiment_advice"], "enzyme_identifiers": []},
        interpret_request("For a PET bench screen, point me toward a useful starting panel."),
    )
    assert "multi_study" in request["objectives"]
    names = {t["function"]["name"] for t in select_tool_catalog(request)}
    assert {"investigate_pet_evidence", "discover_pet_enzymes", "advise_next_experiment"} <= names
    assert names <= {
        "investigate_pet_evidence",
        "discover_pet_enzymes",
        "advise_next_experiment",
        "load_nrel",
        "parse_nrel",
        "filter_observations",
        "assess_evidence",
        "check_comparability",
    }
    assert not {"search_pazy_pet", "get_pazy_metadata", "get_associated_literature"} & names


def test_nrel_only_review_with_exact_conditions_executes_local_tools_without_semantic_plan():
    question = "Using NREL only, investigate DP043 at 40C and 60C, pH 6.5, crystalline PET powder."
    view = run_offline(question)["result"]
    assert view["observations"] and {o["enzyme_identifier"] for o in view["observations"]} == {
        "DP043"
    }
    assert {o["temperature_c"] for o in view["observations"]} == {40.0, 60.0}
    assert view["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert view["source_coverage"][0]["status"] == "FOUND"
    assert view["source_coverage"][1]["status"] == "EXCLUDED"
    assert not view.get("multi_study")


def test_semantic_planning_receives_policy_and_cannot_invent_an_exact_advisor_pair():
    from zymecycle.agent_request import PLANNING_POLICY, select_tool_catalog

    q = "For a PET bench screen, point me toward a useful starting panel."
    backend = NativeScript(
        [("discover_pet_enzymes", {}), ("advise_next_experiment", {"enzyme_identifier": "ESM050"})],
        plan={"objectives": ["enzyme_discovery", "experiment_advice"], "enzyme_identifiers": []},
    )
    report = run_investigation(q, backend)["result"]
    assert any(
        m["role"] == "system" and m["content"] == PLANNING_POLICY for m in backend.messages[-1]
    )
    assert report["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert report["communication"]["status"] == "VERIFIED_EXTRACTIVE"
    request = validate_research_plan(backend.plan, interpret_request(q))
    advisor = next(
        t["function"]
        for t in select_tool_catalog(request)
        if t["function"]["name"] == "advise_next_experiment"
    )
    assert not {"condition_a", "condition_b"} & advisor["parameters"]["properties"].keys()
    a = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
    rejected = run_investigation(
        q,
        NativeScript(
            [
                ("discover_pet_enzymes", {}),
                (
                    "advise_next_experiment",
                    {
                        "enzyme_identifier": "ESM050",
                        "condition_a": a,
                        "condition_b": {**a, "temperature_c": 60},
                    },
                ),
            ],
            plan=backend.plan,
        ),
    )["result"]
    assert rejected["trace"][-1]["error_code"] == "ARGUMENT_SCHEMA"
    assert rejected["synthesis"]["status"] == "INCOMPLETE"


def test_requested_exact_pair_remains_available_to_the_scientific_advisor():
    from zymecycle.agent_request import select_tool_catalog

    q = "What experiment should I perform next for ESM050 at 40C versus 60C, pH 6.5, crystalline PET powder using NREL and PAZy?"
    advisor = next(
        t["function"]
        for t in select_tool_catalog(interpret_request(q))
        if t["function"]["name"] == "advise_next_experiment"
    )
    assert {"condition_a", "condition_b"} <= advisor["parameters"]["properties"].keys()
    assert {"condition_a", "condition_b"} <= set(advisor["parameters"]["required"])


@pytest.mark.parametrize(
    "question",
    [
        "Compare ESM050 on polypropylene powder at 40C and 60C, pH 6.5.",
        "Compare kcat/KM of ESM050 on PET powder at 40C and 60C, pH 6.5.",
    ],
)
def test_semantic_planning_label_cannot_hide_explicit_scientific_scope_errors(question):
    report = run_investigation(
        question,
        NativeScript([("investigate_pet_evidence", {"query": {"protein_name": "ESM050"}})]),
    )["result"]
    assert report["validation"]["scientific_scope"]["issues"]
    report["routing"]["objectives"].append("exploratory_planning")
    report["validation"]["issues"].append(
        "PLANNING_REQUIRES_RESEARCHER_CONSTRAINTS: candidate conditions need clarification; no established optimum or completed experimental plan is claimed."
    )
    result = validate_explanation({"render": "CANONICAL_SELECTED"}, report)
    assert result["status"] == "DETERMINISTIC_FALLBACK"
    assert report["synthesis"]["status"] == "INCOMPLETE"
