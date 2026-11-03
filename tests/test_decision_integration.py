"""Mocked model transport plus actual local evidence; no claim of Muse acceptance."""

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from test_decision import QUESTION, option

from petzyme.agent_types import AgentState
from petzyme.dialogue import prepare_dialogue_turn, validate_dialogue_context
from petzyme.meta import MODEL, MetaModelClient
from petzyme.ui_adapter import investigate_with_backend


class PlanningTransport:
    def __init__(self, continuation=False, bad=None):
        self.continuation = continuation
        self.bad = bad
        self.requests = []

    def respond(self, request):
        p = json.loads(request.content)
        self.requests.append(p)
        assert p["max_completion_tokens"] == 10000 and p["tool_choice"] == "auto"
        texts = " ".join(m.get("content") or "" for m in p["messages"])
        assert "sequence_sha256" not in texts and "/Users/" not in texts
        replies = [json.loads(m["content"]) for m in p["messages"] if m["role"] == "tool"]
        if not self.continuation and not replies:
            args = [("triage_protein", {"name": n}) for n in ["ESM050", "LCC-ICCG"]]
        elif not self.continuation and len(replies) == 2:
            args = [
                (
                    "investigate_pet_evidence",
                    {"query": {"protein_name": n, "ph": 6.5, "form": "powder"}, "limit": 3},
                )
                for n in ["ESM050", "LCC-ICCG"]
            ]
        else:
            args = []
        if args:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{len(replies) + i}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(a)},
                    }
                    for i, (name, a) in enumerate(args)
                ],
            }
            finish = "tool_calls"
        else:
            if self.continuation:
                assert "retained_nrel_" in texts and "previous_alternatives" in texts
                # Source IDs come from the automatic revalidation envelope, never invented records.
                meta = json.loads(p["messages"][1]["content"].split("not retained. ", 1)[1])
                facts = meta["retained_source_evidence"]["facts"]
                ids = meta["required_fact_ids"]
            else:
                facts = [f for r in replies for f in r["facts"]]
                ids = replies[-1]["final_selection"]["required_fact_ids"]
            s = AgentState("unused")
            s.facts = facts
            selected = {
                "action": "EXPERIMENTAL_DECISION",
                "alternatives": [option(s), option(s, "P2", "MATCHED_ENZYME_COMPARISON")],
                "ranking": ["P2", "P1"] if self.continuation else ["P1", "P2"],
                "rationale": [
                    {
                        "preferred": "P2" if self.continuation else "P1",
                        "over": "P1" if self.continuation else "P2",
                        "criterion": "EQUIPMENT_FEASIBILITY"
                        if self.continuation
                        else "TEMPERATURE_COVERAGE",
                        "evidence_ids": ids,
                    }
                ],
                "fact_ids": ids,
            }
            if self.bad == "facts_only":
                selected = {"fact_ids": ids}
            if self.bad == "winner":
                selected["winner"] = "ESM050"
            if self.bad == "ids":
                selected["fact_ids"] = []
            message = {"role": "assistant", "content": json.dumps(selected)}
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "total_tokens": 180,
                    "completion_tokens_details": {"reasoning_tokens": 40},
                },
            },
        )


def run(payload, backend):
    return investigate_with_backend(
        "Local experimental decision",
        MetaModelClient(
            transport=httpx.MockTransport(backend.respond),
            completion_tokens=10000,
            max_estimated_cost_usd=10,
        ),
        dialogue_input=payload,
    )


def first_payload():
    return {"context": None, "text": QUESTION, "input_kind": "ANSWER", "decision_support": True}


def test_actual_two_turn_agent_preserves_facts_and_changes_experiment_under_new_constraint():
    b = PlanningTransport()
    first = run(first_payload(), b)["result"]
    assert first["synthesis"]["status"] == "VALIDATED_EXPERIMENTAL_DECISION"
    assert (
        first["dialogue"]["outcome"] == "EXPERIMENTAL_DECISION"
        and first["dialogue"]["decision"]["selected_id"] == "P1"
    )
    c = first["dialogue"]["context"]
    assert (
        c["assay_ph"] is None
        and c["maximum_ph"] == 7
        and len(c["decision"]["nrel_references"]) == 4
    )
    assert validate_dialogue_context(c)
    assert "original_text" not in json.dumps(c) and "protein_id" not in json.dumps(c)
    follow = {
        "context": c,
        "text": "I can only run experiments at 40C now.",
        "input_kind": "ANSWER",
    }
    next_backend = PlanningTransport(continuation=True)
    second = run(follow, next_backend)["result"]
    assert second["synthesis"]["status"] == "VALIDATED_EXPERIMENTAL_DECISION"
    d = second["dialogue"]
    assert d["context"]["temperatures_c"] == [40] and d["context"]["assay_ph"] is None
    assert (
        d["decision"]["selected_id"] == "P2"
        and d["decision"]["previous_option_status"]["P1"] == "BLOCKED_BY_UPDATED_TEMPERATURES"
    )
    assert d["context"]["model_attempts"] == 4 and d["context"]["tool_attempts"] == 4
    assert (
        len(d["decision"]["evidence_facts"]) == 4
        and d["context"]["usage"]["actual_http_requests"] == 4
    )
    assert second["selection_counts"]["omitted"] == 0
    assert (
        second["workflow_status"] == "PARTIAL"
        and next_backend.requests[0]["messages"][0]["role"] == "system"
    )


@pytest.mark.parametrize("bad", ["winner", "ids", "facts_only"])
def test_invalid_model_decision_is_rejected_without_canned_recommendation(bad):
    r = run(first_payload(), PlanningTransport(bad=bad))["result"]
    assert r["synthesis"]["status"] == "INCOMPLETE"
    assert (
        "decision" not in r["dialogue"]
        and r["dialogue"]["context"]["decision"]["previous_alternatives"] == []
    )


def test_unrelated_new_investigation_has_no_prior_facts_or_choices():
    _, c, _ = prepare_dialogue_turn(first_payload())
    assert not c["decision"]["nrel_references"] and not c["decision"]["previous_alternatives"]
    with pytest.raises(ValueError):
        prepare_dialogue_turn(
            {"context": c, "text": "I can only run experiments at 80C now.", "input_kind": "ANSWER"}
        )


def test_real_app_callbacks_display_hypothetical_choices_and_adapt_without_caching_results(
    monkeypatch,
):
    calls = []

    def fake(question, **kwargs):
        calls.append(1)
        payload = kwargs["dialogue_input"]
        return run(payload, PlanningTransport(continuation=payload["context"] is not None))

    monkeypatch.setattr("petzyme.ui_dialogue.run_keychain_investigation", fake)
    at = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py", default_timeout=30
    ).run()
    at.sidebar.radio[0].set_value("Live Muse dialogue").run()
    at.checkbox(key="dialogue_decision_support").check()
    at.text_area(key="dialogue_text").input(QUESTION)
    at.checkbox(key="dialogue_authorize_0").check()
    at.button(key="dialogue_send").click().run(timeout=60)
    assert not at.exception and any(
        "P1" in s.value and "HYPOTHETICAL" in s.value for s in at.success
    )
    assert any("not live Muse acceptance" in c.value for c in at.caption)
    at.text_area(key="dialogue_text").input("I can only run experiments at 40C now.")
    at.checkbox(key="dialogue_authorize_1").check()
    at.button(key="dialogue_send").click().run(timeout=60)
    assert not at.exception and any("P2" in s.value for s in at.success)
    assert at.session_state["live_dialogue_context"]["temperatures_c"] == [40]
    assert "view" not in at.session_state
    at.run()
    assert len(calls) == 2 and not at.success
    at.button(key="dialogue_reset").click().run()
    assert "live_dialogue_context" not in at.session_state


def test_equipment_subsets_do_not_open_a_general_numeric_vector_channel():
    from petzyme.privacy import outbound_safe

    allowed = {"temperatures_c": [40, 60]}
    assert outbound_safe({"temperatures_c": [40]}, approved_subset_arrays=allowed)
    assert not outbound_safe({"temperatures_c": [80]}, approved_subset_arrays=allowed)
    assert not outbound_safe({"embedding": [40]}, approved_subset_arrays=allowed)


def test_decision_ui_distinguishes_controls_and_unknown_feasibility_without_changing_science():
    from copy import deepcopy

    result = run(first_payload(), PlanningTransport())["result"]
    decision = result["dialogue"]["decision"]
    original = deepcopy(decision)
    selected = next(
        a for a in decision["alternatives"] if a["proposal"]["option_id"] == decision["selected_id"]
    )
    assert selected["resource_feasibility"]["status"] == "UNKNOWN"
    assert (
        "Availability of PET_MATERIAL" in selected["resource_feasibility"]["unknown_requirements"]
    )

    def render(payload):
        from petzyme.ui_dialogue import render_experimental_decision

        render_experimental_decision(payload, "MOCK_META")

    at = AppTest.from_function(render, args=(decision,), default_timeout=30).run()
    assert not at.exception and not at.error
    assert any("Conditional next experiment" in element.value for element in at.success)
    assert any(
        "feasibility is UNKNOWN" in element.value
        and "substrate availability" in element.value
        and "assay calibration" in element.value
        for element in at.warning
    )
    texts = [element.value for element in at.text]
    assert "Experimental controls: substrate blank, reference control" in texts
    assert "Replication design: independent replication" in texts
    assert "Analytical validation: measurement calibration, detection limit" in texts
    assert "Uncertainty/statistical reporting: uncertainty" in texts
    assert not any(text.startswith("Required controls:") for text in texts)
    assert "Unverified resource requirement: Availability of PET_MATERIAL" in texts
    assert decision == original  # Presentation must not alter validated scientific results.
