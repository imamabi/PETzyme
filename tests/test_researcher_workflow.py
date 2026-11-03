"""Real local evidence, synthetic providers and independent workflow/privacy checks."""

import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_scientific_scope import ENDPOINT, KINETIC, POLYPROPYLENE, AllPetFacts

from scripts.audit_nrel_units import read_activity_workbook, reconcile_nrel
from zymecycle.agent import run_investigation
from zymecycle.agent_request import interpret_request, required_tool_steps, select_tool_catalog
from zymecycle.runtime_config import RuntimeLimits, parse_runtime_limits
from zymecycle.ui_adapter import investigate_with_backend
from zymecycle.unit_context import unit_context

QUESTION = "I want to perform experiment for ESM050, what are the recommended conditions for optimal performance?"


class PlanningBackend:
    mode = "MOCK_META"

    def __init__(self):
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(deepcopy(messages))
        replies = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        steps = [
            ("load_nrel", {}),
            ("parse_nrel", {"table_ref": "p0:table"}),
            (
                "filter_observations",
                {"observations_ref": "p1:observations", "enzyme_identifier": "ESM050"},
            ),
            ("assess_evidence", {"observations_ref": "p2:observations"}),
        ]
        if len(replies) < len(steps):
            name, args = steps[len(replies)]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "p" + str(len(replies)),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        coverage = [json.loads(m['content'][m['content'].index('{'):]) for m in messages
                    if m['role']=='system' and m['content'].startswith('Application coverage safeguard')]
        return {
            "role": "assistant",
            "content": json.dumps({"fact_ids": coverage[-1]['final_selection']['required_fact_ids'] if coverage else [f["id"] for r in replies for f in r["facts"]]}),
        }


def test_open_planning_is_not_a_silently_assumed_quantitative_comparison():
    req = interpret_request(QUESTION)
    assert "exploratory_planning" in req["objectives"]
    assert req["scientific_scope"]["material"] == "UNSPECIFIED"
    assert req["temperatures"] == req["ph_values"] == req["pet_forms"] == []
    # The local cross-source composition is required: live evidence shows an optional
    # composition is never selected and broad planning otherwise completes NREL-only.
    assert required_tool_steps(req) == {
        "load_nrel",
        "parse_nrel",
        "filter_observations",
        "assess_evidence",
        "investigate_pet_evidence",
    }
    assert "assess_evidence" in {t["function"]["name"] for t in select_tool_catalog(req)}


def test_real_local_planning_has_valid_selection_and_truthful_partial_clarification():
    backend = PlanningBackend()
    result = investigate_with_backend(QUESTION, backend)
    view = result["result"]
    assert result["status"] == "PARTIAL" and view["workflow_status"] == "PARTIAL"
    assert view["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert "Planning clarification:" in view["planning_question"]
    assert view['multi_study']['nrel']['returned'] and view['multi_study']['pazy']['returned']
    assert {r['source']:r['status'] for r in view['source_coverage']} == {'NREL':'FOUND','PAZy':'FOUND'}
    assert "No publishable" not in view["headline"]
    assert not view["scientific_scope"]["background_only"]
    assert view["scientific_scope"]["request"]["material"] == "UNSPECIFIED"
    assert all(o["unit_status"] == "UNKNOWN/UNVERIFIED" for o in view["observations"])
    outbound = json.dumps(backend.requests)
    assert "experimentally established optimum" in outbound
    for restricted in ("sequence_sha256", "amino_acid_sequence", "embedding_vector", "/Users/"):
        assert restricted not in outbound


@pytest.mark.parametrize(
    "question",
    [
        "Which candidate conditions should I plan for DP038?",
        "Recommend experiments to optimize LCC-ICCG conditions.",
        "What conditions would optimize an experiment for ESM050?",
        "Investigate DP038. What experimental conditions and outcomes are reported, and what should I test next?",
        "What should we try next for LCC-ICCG?",
        "What to measure next for ESM050?",
        "Suggest follow-up experiments for DP043.",
    ],
)
def test_general_planning_cues_are_not_enzyme_specific(question):
    assert "exploratory_planning" in interpret_request(question)["objectives"]


@pytest.mark.parametrize("enzyme", ["ESM050", "DP043", "LCC-ICCG"])
def test_broad_investigation_requires_cross_source_composition(enzyme):
    req = interpret_request(
        f"Investigate {enzyme}. What experimental conditions and outcomes are reported, "
        "and what should I test next?"
    )
    assert "exploratory_planning" in req["objectives"]
    assert "investigate_pet_evidence" in required_tool_steps(req)
    assert "investigate_pet_evidence" in {t["function"]["name"] for t in select_tool_catalog(req)}


def test_broad_investigation_view_combines_sources_with_planning_context():
    class CompletePlanning(PlanningBackend):
        def complete(self, messages, tools):
            replies = [m for m in messages if m["role"] == "tool"]
            if len(replies) == 4:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "p4",
                            "type": "function",
                            "function": {
                                "name": "investigate_pet_evidence",
                                "arguments": json.dumps({"query": {"protein_name": "ESM050"}, "limit": 1}),
                            },
                        }
                    ],
                }
            return super().complete(messages, tools)

    view = investigate_with_backend(QUESTION, CompletePlanning())["result"]
    assert view["multi_study"] and view["observations"]
    assert view["multi_study"]["nrel"]["observations"]
    assert view["multi_study"]["pazy"]["records"]
    assert view["scientific_scope"]["request"]["purpose"] == "EXPERIMENT_PLANNING"
    assert "Planning clarification:" in view["planning_question"]


def test_explicit_single_source_questions_keep_their_scope():
    nrel_only = interpret_request("Investigate ESM050 using NREL data only. What was measured?")
    assert nrel_only["sources"] == ["NREL"]
    assert "investigate_pet_evidence" not in required_tool_steps(nrel_only)
    pazy_only = interpret_request("What PAZy evidence exists for ESM050? PAZy only.")
    assert pazy_only["sources"] == ["PAZy"]
    names = {t["function"]["name"] for t in select_tool_catalog(pazy_only)}
    assert "load_nrel" not in names and "search_pazy_pet" in names


@pytest.mark.parametrize(
    "question",
    [
        POLYPROPYLENE,
        KINETIC,
        "Recommend optimal experiment conditions on polypropylene for ESM050.",
        "Recommend kcat/KM comparison conditions on PET for ESM050.",
    ],
)
def test_scope_repair_survives_planning_language(question):
    report = run_investigation(question, AllPetFacts())["result"]
    assert not report["validation"]["complete"]
    assert report["validation"]["scientific_scope"]["issues"]
    assert report["validation"]["scientific_scope"]["background_only"]


def test_supported_recorded_pet_comparison_and_values_unchanged():
    report = run_investigation(ENDPOINT, AllPetFacts())["result"]
    assert report["validation"]["complete"]
    measured = [f["text"] for f in report["facts"] if f["evidence_category"] == "MEASURED"]
    assert any("0.08499999999999999" in v for v in measured)
    assert any("0.13" in v for v in measured)


def test_attempt_defaults_and_config_are_finite_and_generous():
    d = RuntimeLimits()
    assert (d.turn_models, d.turn_tools, d.session_models, d.session_tools) == (16, 16, 96, 128)
    assert parse_runtime_limits(d.values()) == d
    assert RuntimeLimits(turn_models=32, turn_tools=32).turn_models == 32
    with pytest.raises(ValueError):
        RuntimeLimits(turn_models=33)


def test_more_than_six_real_scientific_steps_can_finish_without_relaxed_grounding():
    class Longer(AllPetFacts):
        def complete(self, messages, tools):
            replies = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            a = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
            steps = [
                ("load_nrel", {}),
                ("parse_nrel", {"table_ref": "x0:table"}),
                (
                    "filter_observations",
                    {"observations_ref": "x1:observations", "enzyme_identifier": "ESM050"},
                ),
                ("assess_evidence", {"observations_ref": "x2:observations"}),
                ("check_comparability", {"observations_ref": "x2:observations"}),
                (
                    "investigate_conditions",
                    {
                        "observations_ref": "x1:observations",
                        "enzyme_identifier": "ESM050",
                        "condition_a": a,
                        "condition_b": {**a, "temperature_c": 60},
                    },
                ),
                ("identify_evidence_gaps", {"investigation_ref": "x5:investigation"}),
            ]
            if len(replies) < len(steps):
                n, args = steps[len(replies)]
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x" + str(len(replies)),
                            "type": "function",
                            "function": {"name": n, "arguments": json.dumps(args)},
                        }
                    ],
                }
            return {
                "role": "assistant",
                "content": json.dumps({"fact_ids": [f["id"] for r in replies for f in r["facts"]]}),
            }

    question = ENDPOINT + " What information is missing?"
    backend = Longer()
    report = run_investigation(question, backend)["result"]
    assert report["model_requests"] == 8 and report["tool_attempts"] == 7
    assert all(e["status"] == "SUCCESS" for e in report["trace"])
    assert report["validation"]["grounded"] and report["validation"]["complete"]
    assert report["validation"]["selection_counts"]["omitted"] == 0


def test_request_limit_is_distinct_from_completion_and_scope_failures():
    report = run_investigation(QUESTION, PlanningBackend(), max_model_requests=2)["result"]
    assert report["synthesis"]["reason_code"] == "MODEL_REQUEST_LIMIT"
    assert report["model_requests"] == 2 and not report["validation"]["complete"]


@pytest.mark.parametrize("bad", ["missing_id", "unsupported_optimum", "invented_condition"])
def test_planning_never_weakens_grounding_or_accepts_unspecified_conditions(bad):
    class Invalid(PlanningBackend):
        def complete(self, messages, tools):
            answer = super().complete(messages, tools)
            if bad == "invented_condition" and answer.get("tool_calls"):
                call = answer["tool_calls"][0]
                if call["function"]["name"] == "filter_observations":
                    args = json.loads(call["function"]["arguments"])
                    args["temperature_c"] = 60
                    call["function"]["arguments"] = json.dumps(args)
            if not answer.get("tool_calls"):
                value = json.loads(answer["content"])
                if bad == "missing_id":
                    value["fact_ids"] = [i for i in value["fact_ids"] if i != "p2.f1"]
                if bad == "unsupported_optimum":
                    value["established_optimum"] = "60C"
                answer["content"] = json.dumps(value)
            return answer

    report = run_investigation(QUESTION, Invalid())["result"]
    assert report["synthesis"]["status"] == "INCOMPLETE"
    assert not report["validation"]["complete"]


def test_spending_guard_has_a_separate_safe_status():
    from zymecycle.agent_types import ReasoningError

    class Blocked:
        mode = "MOCK_META"
        budget_blocked = True

        def complete(self, messages, tools):
            raise ReasoningError("Safe failure")

    report = run_investigation(QUESTION, Blocked())["result"]
    assert report["synthesis"]["reason_code"] == "SPENDING_LIMIT"
    assert "no further request" in report["synthesis"]["reason"]


def test_live_ui_configuration_applies_before_launch_and_reruns_do_not_repeat(monkeypatch):
    calls = []

    def fake(question, **kwargs):
        calls.append(kwargs)
        return investigate_with_backend(
            question, PlanningBackend(), runtime_limits=kwargs["runtime_limits"]
        )

    monkeypatch.setattr("zymecycle.ui_live.run_keychain_investigation", fake)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py", default_timeout=30
    ).run()
    app.radio(key="mode").set_value("Live Meta agent").run()
    app.text_area(key="question").set_value(QUESTION)
    app.number_input(key="oneshot_turn_models").set_value(24)
    app.number_input(key="oneshot_turn_tools").set_value(20)
    app.checkbox[0].check()
    app.button(key="investigate").click().run()
    assert not app.exception and len(calls) == 1
    assert calls[0]["runtime_limits"].turn_models == 24
    assert calls[0]["runtime_limits"].turn_tools == 20
    assert calls[0]["completion_tokens"] == 10000
    assert calls[0]["cost_ceiling_usd"] == 10
    ledger = app.session_state["runtime_ledger"]
    assert ledger["model_attempts"] == 6 and ledger["tool_attempts"] == 5
    app.run()
    assert len(calls) == 1


@pytest.mark.parametrize("manifest", [None, {}, {"unit_reconciliation": {"paper_doi": "bad"}}])
def test_missing_unit_documentation_never_verifies_csv(manifest):
    result = unit_context(manifest)
    assert "unavailable" in result["publication"]
    assert "UNKNOWN/UNVERIFIED" in result["record_verification"]
    assert "No automatic" in result["csv_inference"]


def test_publication_workbook_inference_and_record_status_stay_distinct():
    result = unit_context(json.loads(Path("NREL_SOURCE_MANIFEST.json").read_text()))
    assert "48 h" in result["publication"] and "µmol" in result["publication"]
    assert "mM" in result["workbook"] and "mL" in result["workbook"]
    assert "hypothesis" in result["csv_inference"]
    assert "UNKNOWN/UNVERIFIED" in result["record_verification"]


def workbook_fixture():
    headers = [
        "sample_id",
        "pH",
        "temperature_c",
        "substrate",
        "reaction_vol_mL",
        "mM_product_per_mg_enzyme",
        "mmol_product_per_mg_enzyme",
        "umol_product_per_mg_enzyme",
    ]
    columns = ["A", "B", "C", "E", "D", "U", "V", "W"]

    def cell(col, row, value):
        return (
            f'<c r="{col}{row}" t="inlineStr"><is><t>{value}</t></is></c>'
            if isinstance(value, str)
            else f'<c r="{col}{row}"><v>{value}</v></c>'
        )

    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1">'
        + "".join(cell(c, 1, h) for c, h in zip(columns, headers))
        + '</row><row r="2">'
        + "".join(
            cell(c, 2, v)
            for c, v in zip(columns, ["ESM050", 6.5, 40, "cryPow", 0.5, 159.11, 0.08, 80])
        )
        + "</row></sheetData></worksheet>"
    )
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        z.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>',
        )
        z.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Activity" sheetId="1" r:id="r1"/></sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return data.getvalue()


def test_actual_xlsx_parser_uses_cached_values_not_invented_formulas():
    rows, meta = read_activity_workbook(workbook_fixture())
    assert rows[0]["sample_id"] == "ESM050" and rows[0]["mM_product_per_mg_enzyme"] == 159.11
    assert meta["formula_count"] == 0 and meta["formula_examples"] == {}
    assert meta["column_headers"]["W"] == "umol_product_per_mg_enzyme"
    audit = reconcile_nrel(",activity_at_6.5_40_cryPow\nESM050,0.08\n", rows)
    assert audit["counts"]["reconstructed_matches"] == 1
    assert audit["unit_status"] == "UNKNOWN/UNVERIFIED"


def test_planning_ui_clarification_and_four_unit_levels_with_real_evidence():
    source = 'from test_researcher_workflow import QUESTION,PlanningBackend\nfrom zymecycle.ui_adapter import investigate_with_backend\nfrom streamlit_app import render_findings\nrender_findings(investigate_with_backend(QUESTION,PlanningBackend())["result"])'
    app = AppTest.from_string(
        'import sys\nsys.path.insert(0,"tests")\n' + source, default_timeout=30
    ).run()
    assert not app.exception
    assert any("Planning clarification" in m.value for m in app.info)
    assert not any(
        "Scientific scope: evidence below is background only" in m.value for m in app.warning
    )
    assert any("Publication:" in m.value for m in app.caption)


def test_runtime_controls_and_aggregate_ledger_survive_resets_without_records():
    source = """
import streamlit as st
from zymecycle.ui_runtime import runtime_controls,remaining_limits,runtime_ledger
from zymecycle.ui_dialogue import clear_live_dialogue
limits=runtime_controls("check")
st.text(str(remaining_limits(limits)[0].values()))
st.button("Reset",on_click=clear_live_dialogue)
"""
    app = AppTest.from_string(source).run()
    assert not app.exception
    app.number_input(key="check_turn_models").set_value(24).run()
    assert "24" in app.text[0].value
    app.session_state["runtime_ledger"] = {
        "model_attempts": 12,
        "tool_attempts": 4,
        "usage": {
            "actual_http_requests": 12,
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "reasoning_tokens": 0,
            "unknown_usage_requests": 0,
            "unknown_reasoning_requests": 0,
            "reserved_cost_usd": 0.5,
            "elapsed_seconds": 1.0,
        },
    }
    app.button[0].click().run()
    ledger = app.session_state["runtime_ledger"]
    assert ledger["model_attempts"] == 12 and ledger["usage"]["reserved_cost_usd"] == 0.5
    assert set(ledger) == {"model_attempts", "tool_attempts", "usage"}


def test_uncertain_worker_holds_budget_and_known_usage_reconciles(monkeypatch):
    from zymecycle import ui_runtime

    monkeypatch.setattr(ui_runtime.st, "session_state", {})
    limits = RuntimeLimits(session_cost_usd=1)
    available, cost = ui_runtime.remaining_limits(limits)
    reservation = ui_runtime.reserve_live_usage(available, cost)
    assert ui_runtime.runtime_ledger()["usage"]["reserved_cost_usd"] == 1
    with pytest.raises(ValueError, match="SPENDING_LIMIT"):
        ui_runtime.remaining_limits(limits)
    view = investigate_with_backend(QUESTION, PlanningBackend())["result"]
    view["mode"] = "LIVE_META"
    ui_runtime.record_live_usage(view, reservation)
    assert ui_runtime.runtime_ledger()["usage"]["reserved_cost_usd"] == 1
    view["mode"] = "MOCK_META"
    ui_runtime.record_live_usage(view, reservation)
    ledger = ui_runtime.runtime_ledger()
    assert ledger["model_attempts"] == 6 and ledger["tool_attempts"] == 5
    assert ledger["usage"]["reserved_cost_usd"] == 0


def test_cli_uses_explicit_runtime_controls_without_paid_inference(monkeypatch, capsys):
    import scripts.research_agent as cli

    recorded = []

    def run(question, backend, **kwargs):
        recorded.append(
            (backend.completion_tokens, backend.max_estimated_cost_usd, kwargs["runtime_limits"])
        )
        return run_investigation(
            question, PlanningBackend(), runtime_limits=kwargs["runtime_limits"]
        )

    monkeypatch.setattr(cli, "run_investigation", run)
    assert (
        cli.main(
            [
                QUESTION,
                "--model-requests",
                "24",
                "--tool-attempts",
                "20",
                "--completion-tokens",
                "20000",
                "--spending-ceiling-usd",
                "2",
            ]
        )
        == 1
    )
    allowance, cost, limits = recorded[0]
    assert allowance == 20000 and cost == 2 and limits.turn_models == 24 and limits.turn_tools == 20
    assert "PARTIAL" in capsys.readouterr().out


def test_fully_specified_optimization_keeps_exact_condition_advisor():
    req = interpret_request(
        "Optimize experiment conditions for ESM050 on crystalline PET powder at 40C and 60C, pH 6.5."
    )
    assert "experiment_advice" in req["objectives"]
    assert "recommend_next_experiments" in required_tool_steps(req)


def test_canonical_planning_explanation_accepts_only_the_clarification_state():
    from test_communication import draft_from_facts

    from zymecycle.communication import validate_explanation

    class CompletePlanning(PlanningBackend):
        # The cross-source composition is required for broad planning; the scripted
        # plan must execute it or the draft cannot completely ground required facts.
        def complete(self, messages, tools):
            replies = [m for m in messages if m["role"] == "tool"]
            if len(replies) == 4:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "p4",
                            "type": "function",
                            "function": {
                                "name": "investigate_pet_evidence",
                                "arguments": json.dumps(
                                    {"query": {"protein_name": "ESM050"}, "limit": 1}
                                ),
                            },
                        }
                    ],
                }
            return super().complete(messages, tools)

    report = run_investigation(QUESTION, CompletePlanning())["result"]
    draft = draft_from_facts(report["facts"])
    result = validate_explanation(draft, report)
    assert result["status"] == "VERIFIED_EXTRACTIVE"
    assert report["status"] == "PARTIAL" and not report["validation"]["complete"]
    assert any("Planning clarification:" in s["text"] for s in result["sections"]["uncertainty"])
    broken = deepcopy(report)
    broken["validation"]["issues"].append("SCIENTIFIC_SCOPE_QUANTITY_MISMATCH")
    assert validate_explanation(draft, broken)["status"] == "DETERMINISTIC_FALLBACK"


@pytest.mark.parametrize(
    "mode", ["invented_optimum", "missing_citation", "missing_source", "hidden_clarification"]
)
def test_planning_explanation_does_not_relax_scientific_or_citation_rules(mode):
    from test_communication import draft_from_facts

    from zymecycle.communication import validate_explanation

    report = run_investigation(QUESTION, PlanningBackend())["result"]
    draft = draft_from_facts(report["facts"])
    if mode == "invented_optimum":
        draft["evidence"][0]["text"] = "60C is the experimentally established optimum."
    if mode == "missing_citation":
        draft["evidence"][0].pop("fact_id")
    if mode == "missing_source":
        draft["evidence"].pop()
    if mode == "hidden_clarification":
        draft["uncertainty"] = [
            s for s in draft["uncertainty"] if not s["text"].startswith("Planning clarification:")
        ]
    assert validate_explanation(draft, report)["status"] == "DETERMINISTIC_FALLBACK"
