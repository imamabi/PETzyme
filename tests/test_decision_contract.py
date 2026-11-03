"""Structural reliability and one correction, mocked HTTP with actual local evidence."""

import json
from typing import get_args

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from test_decision import option, selection
from test_decision import state as actual_state
from test_decision_integration import PlanningTransport, first_payload, run

from zymecycle.decision import DecisionError, validate_experimental_decision
from zymecycle.decision_schema import (
    MAX_SCHEMA_ISSUES,
    decision_contract_message,
    decision_json_schema,
    inspect_decision_schema,
    parse_decision_selection,
)
from zymecycle.decision_types import Design, ExperimentOption
from zymecycle.meta import MetaModelClient
from zymecycle.runtime_config import RuntimeLimits
from zymecycle.ui_adapter import investigate_with_backend


@pytest.fixture
def state():
    return actual_state.__wrapped__()


@pytest.mark.parametrize(
    "case,field,code",
    [
        ("identifier", "option_id", "INVALID_IDENTIFIER"),
        ("id_type", "option_id", "WRONG_TYPE"),
        ("design", "design", "UNSUPPORTED_ENUM"),
        ("design_type", "design", "WRONG_TYPE"),
        ("missing", "design", "MISSING_FIELD"),
        ("ph_type", "ph", "WRONG_TYPE"),
        ("temperature_type", "temperatures_c", "WRONG_TYPE"),
    ],
)
def test_precise_value_free_option_errors(state, case, field, code):
    v = selection(state)
    p = v["alternatives"][1]
    if case == "missing":
        del p["design"]
    else:
        p[field] = {
            "identifier": "private-invalid-id",
            "id_type": 42,
            "design": "private-unsupported-design",
            "design_type": [],
            "ph_type": "6.5",
            "temperature_type": "40",
        }[case]
    check = inspect_decision_schema(v)
    assert check["recoverable"] and check["issues"]
    issue = check["issues"][0]
    assert issue["option_index"] == 1 and issue["field"] == field and issue["code"] == code
    assert "private-" not in json.dumps(check)
    assert (issue["received_type"] is None) == (case == "missing")
    with pytest.raises(DecisionError) as error:
        validate_experimental_decision(v, state, [f["id"] for f in state.facts])
    assert error.value.schema_check == check


def test_generated_schema_matches_actual_types_and_example_cannot_ground_itself(state):
    schema = decision_json_schema()
    options = schema["properties"]["alternatives"]
    assert options["minItems"] == 2 and options["maxItems"] == 3
    fields = options["items"]["properties"]
    assert set(fields) == set(ExperimentOption.__annotations__)
    assert fields["design"]["enum"] == list(get_args(Design))
    contract = json.loads(decision_contract_message([40, 60]).split("types: ", 1)[1])
    example = contract["illustrative_structural_example_not_evidence_or_recommendation"]
    assert not inspect_decision_schema(example)["issues"]
    assert parse_decision_selection(example) == example
    with pytest.raises(DecisionError, match="MISSING_OR_UNKNOWN_CANONICAL_IDS"):
        validate_experimental_decision(example, state, [f["id"] for f in state.facts])


def test_diagnostics_cover_each_option_without_reflecting_unknown_keys_or_values(state):
    v = selection(state)
    v["alternatives"].append(option(state, "P3", "MEASUREMENT_QUALIFICATION"))
    for p in v["alternatives"]:
        p["enzymes"] = [123] * 100
        p["private-key-or-sequence"] = "private-value"
        p["design"] = "private-design"
    check = inspect_decision_schema(v)
    assert {i["option_index"] for i in check["issues"]} == {0, 1, 2}
    assert not check["recoverable"] and len(check["issues"]) <= MAX_SCHEMA_ISSUES
    assert "private-" not in json.dumps(check)


@pytest.mark.parametrize("mutation", ["one", "four", "list_type", "nan", "huge"])
def test_counts_types_and_nonfinite_numeric_errors_are_bounded(state, mutation):
    v = selection(state)
    if mutation == "one":
        v["alternatives"] = v["alternatives"][:1]
    elif mutation == "four":
        v["alternatives"] *= 2
    elif mutation == "list_type":
        v["alternatives"] = {}
    else:
        v["alternatives"][0]["ph"] = float("nan") if mutation == "nan" else 10**400
    assert inspect_decision_schema(v)["issues"]


class CorrectionTransport(PlanningTransport):
    def __init__(self, first="identifier", corrected="valid"):
        super().__init__()
        self.first = first
        self.corrected = corrected
        self.finals = 0

    def respond(self, request):
        response = super().respond(request)
        data = response.json()
        message = data["choices"][0]["message"]
        if message.get("tool_calls"):
            return response
        self.finals += 1
        candidate = json.loads(message["content"])
        mode = self.first if self.finals == 1 else self.corrected
        if self.finals > 1:
            payload = json.loads(request.content)
            assert payload["messages"][-1]["role"] == "system"
            instruction = payload["messages"][-1]["content"]
            assert "One explicit output-schema correction" in instruction
            assert "required_fact_ids" in instruction and "expected_structure" in instruction
        if mode == "identifier":
            candidate["alternatives"][0]["option_id"] = "option-one"
        elif mode == "design":
            candidate["alternatives"][0]["design"] = "unknown-design"
        elif mode == "missing":
            del candidate["alternatives"][0]["design"]
        elif mode == "ph":
            candidate["alternatives"][0]["ph"] = 8
        elif mode == "controls":
            candidate["alternatives"][0]["controls"] = []
        elif mode == "grounding":
            candidate["fact_ids"] = []
        elif mode == "winner":
            candidate["winner"] = "ESM050"
        elif mode == "clarification":
            candidate = {
                "action": "ASK_CLARIFICATION",
                "target": "ESM050",
                "request": "CONDITIONS",
                "candidate": None,
                "question": "Which PET form do you want to investigate?",
                "fact_ids": candidate["fact_ids"],
            }
        elif mode == "tools":
            message.update(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "unnecessary",
                            "type": "function",
                            "function": {
                                "name": "triage_protein",
                                "arguments": '{"name":"ESM050"}',
                            },
                        }
                    ],
                }
            )
            data["choices"][0]["finish_reason"] = "tool_calls"
            return httpx.Response(200, json=data)
        message["content"] = json.dumps(candidate)
        return httpx.Response(200, json=data)


@pytest.mark.parametrize("first", ["identifier", "design", "missing"])
def test_one_model_correction_reuses_tools_and_passes_all_scientific_grounding(first):
    backend = CorrectionTransport(first=first)
    view = run(first_payload(), backend)["result"]
    d = view["dialogue"]["decision_contract"]
    assert not d["original_schema_valid"] and not d["original_accepted"]
    assert d["correction_attempts"] == 1 and d["correction_status"] == "ACCEPTED"
    assert d["corrected_accepted"] and d["final_schema_valid"]
    assert view["synthesis"]["status"] == "VALIDATED_EXPERIMENTAL_DECISION"
    assert view["selection_counts"]["omitted"] == 0 and view["workflow_status"] == "PARTIAL"
    assert view["tool_attempts"] == 4 and view["model_requests"] == 4
    assert view["dialogue"]["context"]["model_attempts"] == 4
    assert view["dialogue"]["context"]["usage"]["actual_http_requests"] == 4
    assert "decision_contract" not in view["dialogue"]["context"]
    assert "response_format" not in backend.requests[-1]


@pytest.mark.parametrize(
    "corrected",
    ["identifier", "design", "ph", "controls", "grounding", "winner", "tools", "clarification"],
)
def test_failed_correction_never_gets_a_second_or_a_scientific_substitute(corrected):
    backend = CorrectionTransport(corrected=corrected)
    view = run(first_payload(), backend)["result"]
    d = view["dialogue"]["decision_contract"]
    assert view["synthesis"]["status"] == "INCOMPLETE" and "decision" not in view["dialogue"]
    assert d["correction_attempts"] == 1 and d["correction_status"] == "REJECTED"
    assert not d["original_accepted"] and not d["corrected_accepted"]
    assert backend.finals == 2 and view["tool_attempts"] == 4 and view["model_requests"] == 4
    if corrected == "tools":
        assert view["synthesis"]["reason_code"] == "SCHEMA_CORRECTION_TOOL_CALL_FORBIDDEN"


@pytest.mark.parametrize("first", ["ph", "controls", "grounding", "winner"])
def test_scientific_rejections_are_not_repair_opportunities(first):
    backend = CorrectionTransport(first=first)
    view = run(first_payload(), backend)["result"]
    assert view["synthesis"]["status"] == "INCOMPLETE"
    assert view["dialogue"]["decision_contract"]["correction_attempts"] == 0
    assert backend.finals == 1 and view["model_requests"] == 3


def test_request_and_session_budgets_are_not_extended_for_correction():
    backend = CorrectionTransport()
    client = MetaModelClient(
        transport=httpx.MockTransport(backend.respond), completion_tokens=10000
    )
    view = investigate_with_backend(
        "Local decision",
        client,
        dialogue_input=first_payload(),
        runtime_limits=RuntimeLimits(turn_models=3, session_models=3),
    )["result"]
    assert view["synthesis"]["status"] == "INCOMPLETE"
    assert view["model_requests"] == 3 and backend.finals == 1
    assert view["dialogue"]["decision_contract"]["correction_status"] == "BUDGET_UNAVAILABLE"


def test_pre_http_spending_guard_still_blocks_a_correction():
    backend = CorrectionTransport()
    client = MetaModelClient(
        transport=httpx.MockTransport(backend.respond), completion_tokens=10000
    )
    real_complete = client.complete

    def bounded_complete(messages, tools):
        if backend.finals == 1:
            client.max_estimated_cost_usd = 0.000001
        return real_complete(messages, tools)

    client.complete = bounded_complete
    view = investigate_with_backend("Local decision", client, dialogue_input=first_payload())[
        "result"
    ]
    assert view["synthesis"]["status"] == "INCOMPLETE" and client.budget_blocked
    assert backend.finals == 1 and len(view["provider_telemetry"]) == 3
    assert view["model_requests"] == 4  # Backend attempt, not a fourth HTTP request.


def test_apptest_distinguishes_corrected_success_from_original_output_and_keeps_no_history():
    backend = CorrectionTransport()
    view = run(first_payload(), backend)["result"]

    def render(payload):
        from zymecycle.ui_dialogue import render_dialogue_result

        render_dialogue_result(payload, lambda _: None)

    at = AppTest.from_function(render, args=(view,), default_timeout=30).run()
    assert not at.exception and not at.error
    assert any(
        "Original decision accepted: False" in c.value
        and "schema correction attempts: 1" in c.value
        and "ACCEPTED" in c.value
        for c in at.caption
    )
    assert any("option index 0" in t.value and "INVALID_IDENTIFIER" in t.value for t in at.text)
    rendered = " ".join(t.value for t in at.text)
    assert (
        "option-one" not in rendered
    )  # Rejected values are not shown, only type/field/index metadata.
    assert "decision_contract" not in view["dialogue"]["context"]
    assert "view" not in at.session_state


def test_valid_original_output_is_distinguished_from_success_after_correction():
    backend = PlanningTransport()
    view = run(first_payload(), backend)["result"]
    d = view["dialogue"]["decision_contract"]
    assert d["original_schema_valid"] and d["original_accepted"]
    assert d["final_schema_valid"] and not d["corrected_accepted"]
    assert d["correction_attempts"] == 0 and d["correction_status"] == "NOT_REQUESTED"
    assert d["initial_issues"] == [] and d["final_issues"] == []
