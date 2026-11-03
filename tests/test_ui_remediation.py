"""UI-only failure/state regressions with mocked Meta and real local scientific tools."""

import copy
import json

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from test_decision_integration import PlanningTransport, first_payload, run

from petzyme.agent_evidence import project_evidence, public_tool_update
from petzyme.protein_triage import triage_protein
from petzyme.results import outcome
from petzyme.ui_identity import identity_card


def update_for(result):
    preview, _ = project_evidence("identity_case", "triage_protein", result)
    return public_tool_update(
        {
            "call_id": "identity_case",
            "name": "triage_protein",
            "status": result["status"],
            "arguments": {},
            "refs": {},
            "preview": preview,
            "fact_ids": [],
            "ephemeral": True,
        },
        result,
        1,
        1,
    )


@pytest.mark.parametrize("case", ["SUCCESS", "NO_DATA", "ERROR", "OPTIONAL_MISSING", "INCOMPLETE"])
def test_identity_preview_variants_do_not_invent_fields_or_crash(case):
    result = (
        triage_protein(name="ESM050")
        if case not in ("ERROR", "NO_DATA")
        else (
            outcome("ERROR", None, [], [], None, "Synthetic bounded failure")
            if case == "ERROR"
            else triage_protein(name="unmatched_example_identifier")
        )
    )
    update = update_for(result)
    if case == "OPTIONAL_MISSING":
        for key in ("sequence_validation", "sequence_length", "flagged_symbols", "nrel_candidates"):
            update["data"]["identity"].pop(key, None)
    if case == "INCOMPLETE":
        update["data"]["identity"] = {"status": "SUCCESS"}
    original = copy.deepcopy(update)

    def render(item):
        from petzyme.ui_identity import render_identity_update

        render_identity_update(item)

    at = AppTest.from_function(render, args=(update,), default_timeout=30).run()
    assert not at.exception
    card = identity_card(update)
    assert update == original and card["tool_status"] == result["status"]
    if case == "SUCCESS":
        assert any("CANDIDATE_MATCH" in t.value for t in at.text)
    elif case == "ERROR":
        assert any("ERROR" in w.value for w in at.warning) and not at.text
        assert card["identity_status"] is None
    elif case == "NO_DATA":
        assert result["status"] == "NO_DATA" and any("NO_DATA" in i.value for i in at.info)
    elif case == "OPTIONAL_MISSING":
        assert card["sequence_length"] is None and any(
            "length unavailable" in t.value for t in at.text
        )
    else:
        assert card["identity_status"] is None and any(
            "No identity status was inferred" in w.value for w in at.warning
        )


class FailedTriageTransport(PlanningTransport):
    def __init__(self):
        super().__init__()
        self.failed_call_sent = False

    def respond(self, request):
        response = super().respond(request)
        data = response.json()
        message = data["choices"][0]["message"]
        if not message.get("tool_calls") and not self.failed_call_sent:
            self.failed_call_sent = True
            message.update(
                content=None,
                tool_calls=[
                    {
                        "id": "failed_identity",
                        "type": "function",
                        "function": {
                            "name": "triage_protein",
                            "arguments": json.dumps({"name": "ESM050", "unknown": True}),
                        },
                    }
                ],
            )
            data["choices"][0]["finish_reason"] = "tool_calls"
        return httpx.Response(200, json=data)


def test_failed_triage_can_coexist_with_real_grounded_accepted_decision():
    view = run(first_payload(), FailedTriageTransport())["result"]
    assert view["synthesis"]["status"] == "VALIDATED_EXPERIMENTAL_DECISION"
    failed = next(p for p in view["progress"] if p["call_id"] == "failed_identity")
    assert failed["status"] == "ERROR" and "identity_status" not in failed["data"]["identity"]
    context = copy.deepcopy(view["dialogue"]["context"])

    def render(payload):
        from petzyme.ui_dialogue import render_dialogue_result

        render_dialogue_result(payload, lambda _: None)

    at = AppTest.from_function(render, args=(view,), default_timeout=30).run()
    assert not at.exception and not at.error
    assert any("Conditional next experiment" in s.value for s in at.success)
    assert any("Local identity triage · ERROR" in w.value for w in at.warning)
    assert view["dialogue"]["context"] == context
    assert context["confirmed_labels"] == ["ESM050", "LCC-ICCG"]


def test_normal_page_submission_survives_failed_triage_and_keeps_adaptive_context(monkeypatch):
    from pathlib import Path

    calls = []

    def fake(question, **kwargs):
        payload = kwargs["dialogue_input"]
        calls.append(payload["context"] is not None)
        result = run(
            payload,
            PlanningTransport(continuation=True) if payload["context"] else FailedTriageTransport(),
        )
        for update in result["result"]["progress"]:
            kwargs["on_update"](update)
        return result

    monkeypatch.setattr("petzyme.ui_dialogue.run_keychain_investigation", fake)
    at = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py", default_timeout=30
    ).run()
    at.sidebar.radio[0].set_value("Live Muse dialogue").run()
    at.checkbox(key="dialogue_decision_support").check()
    at.text_area(key="dialogue_text").input(first_payload()["text"])
    at.checkbox(key="dialogue_authorize_0").check()
    at.button(key="dialogue_send").click().run(timeout=60)
    assert not at.exception and not at.error and len(calls) == 1
    assert any("Conditional next experiment" in s.value for s in at.success)
    assert any("Local identity triage · ERROR" in w.value for w in at.warning)
    assert at.checkbox(key="dialogue_decision_support").value is True
    assert at.checkbox(key="dialogue_new").value is False
    assert at.text_area(key="dialogue_text").value == ""
    assert at.session_state["live_dialogue_context"]["confirmed_labels"] == ["ESM050", "LCC-ICCG"]
    assert any(s.label == "Scientific tools finished" for s in at.status)
    at.run()
    assert len(calls) == 1 and not at.success
    assert at.checkbox(key="dialogue_decision_support").value is True
    assert "dialogue_submit_pending" not in at.session_state and "view" not in at.session_state
    at.text_area(key="dialogue_text").input("I can only run experiments at 40C now.")
    at.checkbox(key="dialogue_authorize_1").check()
    at.button(key="dialogue_send").click().run(timeout=60)
    assert not at.exception and not at.error and calls == [False, True]
    context = at.session_state["live_dialogue_context"]
    assert (
        context["temperatures_c"] == [40]
        and context["maximum_ph"] == 7
        and context["assay_ph"] is None
    )
    assert any("Conditional next experiment" in s.value for s in at.success)
    spent = at.session_state["dialogue_spent"]
    at.button(key="dialogue_reset").click().run()
    assert (
        "live_dialogue_context" not in at.session_state
        and at.session_state["dialogue_spent"] == spent
    )
    assert at.checkbox(key="dialogue_decision_support").value is False and len(calls) == 2


def test_readable_experiment_summary_preserves_hypothetical_conditions_and_technical_evidence():
    from petzyme.ui_decision import experiment_summary

    view = run(first_payload(), PlanningTransport())["result"]
    decision = view["dialogue"]["decision"]
    original = copy.deepcopy(decision)
    selected = next(
        a for a in decision["alternatives"] if a["proposal"]["option_id"] == decision["selected_id"]
    )
    summary = experiment_summary(selected["proposal"])
    assert "ESM050 and LCC-ICCG" in summary and "40°C and 60°C" in summary
    assert "proposed pH 6.5" in summary and "crystalline PET powder" in summary
    assert "BALANCED_ENZYME_TEMPERATURE" not in summary

    def render(payload):
        from petzyme.ui_dialogue import render_experimental_decision

        render_experimental_decision(payload, "MOCK_META")

    at = AppTest.from_function(render, args=(decision,), default_timeout=30).run()
    assert not at.exception and not at.error
    assert any("HYPOTHETICAL" in x.value for x in at.success)
    assert any("feasibility is UNKNOWN" in w.value for w in at.warning)
    assert any("What to measure" in m.value for m in at.markdown)
    assert any("Required next action" in m.value for m in at.markdown)
    assert any("0.08499999999999999" in t.value for t in at.text)
    assert any("BALANCED_ENZYME_TEMPERATURE" in t.value for t in at.text)
    assert decision == original


def test_explicit_reset_revokes_unsubmitted_authorization_and_resets_form(monkeypatch):
    from pathlib import Path

    def forbidden(*args, **kwargs):
        raise AssertionError("Reset must not launch an investigation")

    monkeypatch.setattr("petzyme.ui_dialogue.run_keychain_investigation", forbidden)
    at = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py", default_timeout=30
    ).run()
    at.sidebar.radio[0].set_value("Live Muse dialogue").run()
    at.checkbox(key="dialogue_new").uncheck()
    at.checkbox(key="dialogue_decision_support").check()
    at.text_area(key="dialogue_text").input("Unsubmitted identifying information")
    at.checkbox(key="dialogue_authorize_0").check()
    at.session_state["dialogue_spent"] = 0.25
    at.button(key="dialogue_reset").click().run()
    assert not at.exception
    assert at.checkbox(key="dialogue_new").value is True
    assert at.checkbox(key="dialogue_decision_support").value is False
    assert at.text_area(key="dialogue_text").value == ""
    assert at.selectbox(key="dialogue_kind").value == "ANSWER"
    assert at.checkbox(key="dialogue_authorize_1").value is False
    assert "dialogue_authorize_0" not in at.session_state
    assert "live_dialogue_context" not in at.session_state
    assert at.session_state["dialogue_spent"] == 0.25
    at.run()
    assert not at.exception and at.checkbox(key="dialogue_new").value is True
