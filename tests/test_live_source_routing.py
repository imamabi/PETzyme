"""Production-path routing and real local evidence with offline model transport only."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from test_communication import draft_from_facts
from test_communication_integration import expanded_facts

from petzyme.agent import run_investigation
from petzyme.agent_request import interpret_request, select_tool_catalog
from petzyme.communication import validate_explanation
from petzyme.meta import MODEL, MetaModelClient
from petzyme.pet_evidence import investigate_pet_evidence
from petzyme.research_advisor import advise_next_experiment
from petzyme.ui_adapter import investigate_with_backend
from petzyme.ui_live import general_research_question

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Investigate ESM050. What experimental conditions and outcomes are reported, and what should I test next?"


class CombinedScenario:
    """Offline contract fixture, never a claimed autonomous Muse evaluation."""

    def __init__(self, bad=None):
        self.requests = []
        self.bad = bad

    def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        for forbidden in ("/Users/", '"amino_acid_sequence"', '"sequence_sha256"', '"vector"'):
            assert forbidden not in request.content.decode()
        replies = [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]
        if not replies:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "coverage",
                        "type": "function",
                        "function": {
                            "name": "advise_next_experiment",
                            "arguments": json.dumps({"enzyme_identifier": "ESM050"}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            ids = replies[-1]["final_selection"]["required_fact_ids"]
            explanation = draft_from_facts(expanded_facts(payload["messages"]))
            if self.bad == "omit":
                ids = ids[:-1]
            if self.bad == "claim":
                explanation["evidence"][0]["text"] = "60°C proves superior activity."
            message = {
                "role": "assistant",
                "content": json.dumps({"fact_ids": ids, "explanation": explanation}),
            }
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"finish_reason": finish, "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
        )


@pytest.mark.parametrize("enzyme", ["ESM050", "DP043", "LCC-ICCG"])
def test_general_research_routes_without_fabricated_two_enzyme_context(enzyme):
    q = QUESTION.replace("ESM050", enzyme)
    payload = {"context": None, "text": q, "input_kind": "ANSWER"}
    assert general_research_question(payload) == q
    request = interpret_request(q)
    assert request["sources"] == ["NREL", "PAZy"]
    names = {t["function"]["name"] for t in select_tool_catalog(request)}
    assert {"investigate_pet_evidence", "advise_next_experiment"} <= names
    assert not {"search_pazy_pet", "get_pazy_metadata"} & names


def test_structured_dialogue_and_private_input_gates_remain():
    choice = {
        "context": None,
        "text": "Should I use ESM050 or LCC-ICCG? My device supports 40C and 60C, maximum pH 7.",
        "input_kind": "ANSWER",
    }
    assert general_research_question(choice) is None
    assert general_research_question({**choice, "text": "Should I use ESM050 or UNKNOWN?"}) is None
    assert general_research_question({**choice, "input_kind": "SEQUENCE", "text": "ACDEF"}) is None
    with pytest.raises(ValueError):
        general_research_question({**choice, "text": ">private\nACDEFACDEFACDEF"})


def test_explicit_single_source_is_a_pre_execution_boundary(monkeypatch):
    q = "Investigate PAZy-only experimental evidence for ESM050."
    request = interpret_request(q)
    tools = select_tool_catalog(request)
    assert request["sources"] == ["PAZy"] and {t["function"]["name"] for t in tools} == {
        "investigate_pet_evidence", "advise_next_experiment"
    }
    for tool in tools:
        schema = tool["function"]["parameters"]
        assert schema["properties"]["source_scope"]["enum"] == ["PAZY_ONLY"]
        assert "source_scope" in schema["required"]

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit PAZy-only retrieval must not open NREL data")

    monkeypatch.setattr("petzyme.pet_evidence.load_nrel", forbidden)
    result = investigate_pet_evidence({"protein_name": "ESM050"}, source_scope="PAZY_ONLY")
    assert result["result"]["nrel"]["returned"] == 0 and result["result"]["pazy"]["returned"] > 0
    assert all(s["name"] != "NREL" for s in result["provenance"])
    nrel = select_tool_catalog(interpret_request("Investigate NREL-only evidence for ESM050."))
    assert not any(
        "pazy" in t["function"]["name"]
        or t["function"]["name"] in {"investigate_pet_evidence", "advise_next_experiment"}
        for t in nrel
    )


def test_combined_coverage_changes_conditional_proposal_without_inventing_novelty():
    advice = advise_next_experiment("ESM050", planning_goal="VERIFY_POSITIVE")["result"]
    evidence = advice["evidence_results"][0]["result"]
    assert evidence["experimental_coverage"]["pazy_additional_conditions"] == 6
    assert evidence["experimental_coverage"]["missing_in_both_count"] == 0
    assert advice["condition_result"]["result"] is None
    plan = advice["plan_result"]["result"]
    assert plan["basis_kind"] == "EXPERIMENTAL_COVERAGE"
    proposal = next(a for a in plan["recommendations"] if a["category"] == "PROPOSED_EXPERIMENT")
    assert (
        proposal["target_conditions"]
        and "source already documents this condition" in proposal["proposed_action"]
    )
    assert proposal["feasibility"]["status"] == "UNKNOWN" and proposal["prerequisites"]
    assert any(
        "PAZy publication-derived endpoint" in b["statement"] for b in proposal["evidence_basis"]
    )
    target = proposal["target_conditions"][0]
    condition = next(
        c
        for c in evidence["experimental_coverage"]["conditions"]
        if c["temperature_c"] == target["temperature_c"]
        and c["ph"] == target["ph"]
        and c["form"] == "film"
    )
    assert condition["pazy_assays"] > 0 and condition["nrel_observations"] == 0


@pytest.mark.parametrize("bad", [None, "omit", "claim"])
def test_actual_agent_native_science_grounding_and_readable_source_contract(bad):
    scenario = CombinedScenario(bad)
    report = run_investigation(
        QUESTION, MetaModelClient(transport=httpx.MockTransport(scenario.respond))
    )["result"]
    assert report["provider_mode"] == "MOCK_META" and report["status"] == "PARTIAL"
    assert [(e["name"], e["status"]) for e in report["trace"]] == [
        ("advise_next_experiment", "PARTIAL")
    ]
    assert set(report["routing"]["sources"]) == {"NREL", "PAZy"}
    if bad == "omit":
        assert report["validation"]["selection_counts"]["omitted"] == 1
        assert report["synthesis"]["status"] == "INCOMPLETE"
    elif bad == "claim":
        assert report["communication"]["status"] == "DETERMINISTIC_FALLBACK"
        assert report["communication"]["reason_code"] == "UNVERIFIED_WORDING_OR_ROLE"
    else:
        assert report["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
        assert report["validation"]["selection_counts"]["omitted"] == 0
        assert report["communication"]["status"] == "VERIFIED_EXTRACTIVE"
        summaries = [
            x
            for x in report["communication"]["sections"]["evidence"]
            if x["field"] == "measurement_summary"
        ]
        assert len(summaries) == 3 and all(x["sources"] for x in summaries)
        explanation = draft_from_facts(report["facts"])
        explanation["evidence"] = [
            x for x in explanation["evidence"] if x["field"] != "measurement_summary"
        ]
        assert (
            validate_explanation(explanation, report)["reason_code"]
            == "EVIDENCE_COVERAGE_INCOMPLETE"
        )


def test_real_live_muse_ui_path_preserves_both_sources_transiently(monkeypatch):
    scenario = CombinedScenario()
    result = investigate_with_backend(
        QUESTION, MetaModelClient(transport=httpx.MockTransport(scenario.respond))
    )
    assert result["result"]["multi_study"]["pazy"]["records"]
    calls = []

    def launch(question, **kwargs):
        calls.append(kwargs)
        assert kwargs["dialogue_input"]["text"] == QUESTION
        return deepcopy(result)

    monkeypatch.setattr("petzyme.ui_dialogue.run_keychain_investigation", launch)
    app = AppTest.from_file(str(ROOT / "tests/legacy_streamlit_app.py"), default_timeout=30).run()
    app.radio(key="mode").set_value("Live Muse dialogue").run()
    app.text_area(key="dialogue_text").set_value(QUESTION)
    app.checkbox(key="dialogue_authorize_0").set_value(True)
    app.button(key="dialogue_send").click().run()
    assert not app.exception and len(calls) == 1
    combined = next(t.value for t in app.table if "Source" in t.value.columns)
    assert any(combined["Source"] == "PAZy · reported") and any(combined["Source"] == "NREL · inferred conversion")
    assert "live_dialogue_context" not in app.session_state and "view" not in app.session_state
    app.run()
    assert len(calls) == 1 and not app.table


def test_pazy_only_named_evidence_has_executable_offline_intent():
    from petzyme.agent_request import interpret_request
    from petzyme.ui_adapter import run_offline

    question = 'Show PAZy evidence only for ESM050 at 40°C.'
    request = interpret_request(question)
    assert request['sources'] == ['PAZy'] and 'multi_study' in request['objectives']
    view = run_offline(question)['result']
    assert view['multi_study']['pazy']['records']
    assert not view['multi_study']['nrel']['observations']


def test_general_question_crosses_real_launcher_worker_boundary_offline(tmp_path, monkeypatch):
    import inspect
    import os

    from petzyme import ui_live

    for package in ("scripts", "petzyme"):
        directory = tmp_path / package
        directory.mkdir()
        (directory / "__init__.py").write_text(
            "__path__.append(" + repr(str(ROOT / package)) + ")\n"
        )
    common = """import importlib.util, socket, os, sys
from pathlib import Path
ROOT=Path(%r)
def blocked(*a,**k):raise AssertionError('Offline acceptance must not connect')
socket.socket.connect=blocked
""" % str(ROOT)
    (tmp_path / "scripts/launch_meta.py").write_text(
        common
        + """
import subprocess
spec=importlib.util.spec_from_file_location('scripts._launch_real',ROOT/'scripts/launch_meta.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert os.environ.get('MODEL_API_KEY') == 'synthetic-parent-not-forwarded'
module._lookup=lambda password:subprocess.CompletedProcess([],0,b'synthetic-child-only\\n')
raise SystemExit(module.main())
"""
    )
    (tmp_path / "petzyme/ui_live.py").write_text(
        common
        + """
import httpx, json
from test_communication import draft_from_facts
from test_communication_integration import expanded_facts
from petzyme import ui_adapter, meta
MODEL=meta.MODEL
"""
        + inspect.getsource(CombinedScenario)
        + """
assert os.environ['MODEL_API_KEY']=='synthetic-child-only'
scenario=CombinedScenario()
Original=meta.MetaModelClient
meta.MetaModelClient=lambda **kw:Original(transport=httpx.MockTransport(scenario.respond),completion_tokens=kw['completion_tokens'],max_estimated_cost_usd=kw['max_estimated_cost_usd'])
spec=importlib.util.spec_from_file_location('petzyme._live_real',ROOT/'petzyme/ui_live.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
raise SystemExit(module.worker_main())
"""
    )
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), str(ROOT), str(ROOT / "tests")))
    )
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-parent-not-forwarded")
    result = ui_live.run_keychain_investigation(
        "Local identity dialogue",
        authorized=True,
        completion_tokens=10000,
        cost_ceiling_usd=0.5,
        dialogue_input={"context": None, "text": QUESTION, "input_kind": "ANSWER"},
    )
    assert result["result"] is not None
    view = result["result"]
    assert view["submitted_question"] == QUESTION
    assert view["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert view["communication"]["status"] == "VERIFIED_EXTRACTIVE"
    assert len(view["provider_telemetry"]) == 2
    assert view["multi_study"]["pazy"]["returned"] == 3
    assert view["provider_budget"]["limit_usd"] == 0.5
    assert "synthetic-child-only" not in json.dumps(result)
    assert "synthetic-parent-not-forwarded" not in json.dumps(result)
    assert not list(tmp_path.rglob("*.json"))


def test_pazy_only_ranking_does_not_consult_nrel(monkeypatch):
    from petzyme.enzyme_discovery import discover_pet_enzymes

    request = interpret_request("What are the five best PET enzymes using PAZy only?")
    catalog = select_tool_catalog(request)
    tool = next(t["function"] for t in catalog if t["function"]["name"] == "discover_pet_enzymes")
    for definition in catalog:
        if definition["function"]["name"] != "discover_pet_enzymes":
            assert definition["function"]["parameters"]["properties"]["source_scope"]["enum"] == ["PAZY_ONLY"]
    assert request["sources"] == ["PAZy"]
    assert tool["parameters"]["properties"]["include_nrel_references"]["enum"] == [False]
    assert "include_nrel_references" in tool["parameters"]["required"]

    def forbidden(*args, **kwargs):
        pytest.fail("Source-excluded NREL data accessed")

    monkeypatch.setattr("petzyme.enzyme_discovery.load_nrel", forbidden)
    result = discover_pet_enzymes(include_nrel_references=False)
    assert result["result"]["entries"]
    assert all(not e["nrel_references"] for e in result["result"]["entries"])


def test_pazy_only_ui_does_not_claim_two_selected_sources():
    app = AppTest.from_string('''
from petzyme.pet_evidence import investigate_pet_evidence
from petzyme.pet_presenter import preview_pet_evidence
from petzyme.privacy import safe_public_text
from petzyme.ui_experiments import render_combined_experiments
value=investigate_pet_evidence({'protein_name':'IsPETase'},source_scope='PAZY_ONLY')['result']
render_combined_experiments(preview_pet_evidence(value,safe_public_text))
''', default_timeout=30).run()
    assert not app.exception
    text = '\n'.join(m.value for m in app.markdown)
    assert 'lack records in the selected PAZy scope' in text
    assert 'lack records in both selected sources' not in text


def test_documentary_links_do_not_double_count_identical_nrel_measurements():
    from petzyme.pet_accumulation import accumulate_pet_results
    from petzyme.pet_presenter import preview_pet_evidence
    from petzyme.privacy import safe_public_text

    runs = [
        {
            "call_id": str(i),
            "preview": preview_pet_evidence(
                investigate_pet_evidence(q)["result"], safe_public_text
            ),
        }
        for i, q in enumerate([{"protein_name": "ESM050"}, {"protein_name": "ESM050", "ph": 5.5}])
    ]
    value = accumulate_pet_results(
        runs, interpret_request("Compare NREL and PAZy evidence for ESM050.")
    )
    observations = value["nrel"]["observations"]
    assert (
        len(observations) == len({(o["source_row"], o["source_column"]) for o in observations}) == 3
    )
    assert value["accumulation"]["source_conflicts"] == 0
    assert any(o["source_row_links"] for o in observations)
    corrupted = deepcopy(runs)
    corrupted[1]["preview"]["nrel"]["observations"][0]["original_text"] = "999"
    rejected = accumulate_pet_results(
        corrupted, interpret_request("Compare NREL and PAZy evidence for ESM050.")
    )
    assert rejected["accumulation"]["source_conflicts"] > 0


def test_incomplete_muse_keeps_executed_conditional_tool_plan_visible():
    from petzyme.ui_findings import result_overview

    scenario = CombinedScenario()
    view = investigate_with_backend(QUESTION + ' Please replicate a documented positive response.', MetaModelClient(transport=httpx.MockTransport(scenario.respond)))['result']
    view['synthesis'] = {'status': 'INCOMPLETE', 'reason_code': 'COMPLETION_LENGTH_LIMIT'}
    overview = result_overview(view)
    assert 'INCOMPLETE SYNTHESIS' in overview.label
    assert 'Tool-generated conditional proposals' in overview.attribution
    assert any('documented positive response' in step.text for step in overview.steps)
    assert all('UNKNOWN' in step.label for step in overview.steps)


def test_discovery_live_telemetry_is_accessible_without_another_request():
    # Rendering-only fixture: native local data, no live provider invocation.
    source = """
import streamlit as st
from petzyme.ui_adapter import run_local_discovery
from petzyme.ui_results import render_result
v=run_local_discovery('What are the five enzymes with the best response for PET breakdown?')['result']
v['mode']='LIVE_META'
render_result(v,lambda:st.caption('Provider diagnostics available'))
"""
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert any('Provider diagnostics available' in c.value for c in app.caption)
    assert any(e.label == 'Details · complete evidence, sources and execution' for e in app.expander)


def test_pazy_reported_endpoints_count_as_responses_without_establishing_kinetics():
    from petzyme.agent_execution import execute_tool_call
    from petzyme.agent_scope import assess_agent_scope
    from petzyme.agent_types import AgentState

    question = 'Investigate PAZy-only PET experimental evidence for IsPETase outside the NREL publication: show recorded responses.'
    request = interpret_request(question)
    state = AgentState(str(ROOT / 'data/nrel/label_data.csv'))
    call = {'id': 'pazy', 'type': 'function', 'function': {'name': 'investigate_pet_evidence',
            'arguments': json.dumps({'query': {'protein_name': 'IsPETase'}, 'source_scope': 'PAZY_ONLY', 'scope': 'OUTSIDE_NREL_PUBLICATION'})}}
    event = execute_tool_call(call, state, select_tool_catalog(request))
    assert event['status'] == 'PARTIAL'
    scope = assess_agent_scope(request, state)
    assert scope['status'] == 'MATCH' and scope['evidence']['source_names'] == ['PAZy']
    assert 'RECORDED_RESPONSE' in scope['evidence']['quantities']
    kinetic = interpret_request('Compare kcat/KM of IsPETase on PET.')
    rejected = assess_agent_scope(kinetic, state)
    assert 'SCIENTIFIC_SCOPE_QUANTITY_MISMATCH' in rejected['issues']
    assert rejected['background_only']


def test_exact_pazy_scope_requires_both_requested_conditions():
    from petzyme.agent_execution import execute_tool_call
    from petzyme.agent_reporting import validate_investigation
    from petzyme.agent_types import AgentState

    request = interpret_request('Compare PAZy-only recorded PET responses for ESM050 at 40C and 60C, pH 6.5 using crystalline powder.')
    catalog = select_tool_catalog(request)
    assert [t['function']['name'] for t in catalog] == ['investigate_pet_evidence']
    state = AgentState(str(ROOT / 'data/nrel/label_data.csv'))
    def call(temperature):
        return execute_tool_call({'id': 'c' + str(temperature), 'type': 'function', 'function': {
            'name': 'investigate_pet_evidence', 'arguments': json.dumps({'query': {'protein_name': 'ESM050',
                'temperature_c': temperature, 'ph': 6.5, 'form': 'powder', 'morphology': 'crystalline'}, 'source_scope': 'PAZY_ONLY'})}}, state, catalog)
    call(40)
    first = validate_investigation(request, state, {'fact_ids': [f['id'] for f in state.facts]})
    assert 'PAZy-only selection did not cover every requested nominal condition with executed assay records.' in first['issues']
    call(60)
    complete = validate_investigation(request, state, {'fact_ids': [f['id'] for f in state.facts]})
    assert not any('did not cover every requested' in issue for issue in complete['issues'])
    assert all(not e['preview']['nrel']['observations'] for e in state.events)
