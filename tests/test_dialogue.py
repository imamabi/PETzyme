"""Mock Meta decisions and real local triage; never real Muse acceptance."""

import hashlib
import json
from copy import deepcopy

import httpx
import pytest
from test_build10_acceptance import (
    AMBIGUOUS,  # Reuse the existing synthetic input fixture; no new sequence copy.
)

from zymecycle.dialogue import prepare_dialogue_turn, validate_dialogue_context
from zymecycle.meta import MODEL, MetaModelClient
from zymecycle.ui_adapter import investigate_with_backend

QUESTION = "Should I use ESM050 or LCCMC? My device supports 40C and 60C, but I cannot exceed pH 7."


class DialogueTransport:
    def __init__(self, names=None, *, sequence=False, target="LCCMC", candidate=None, bad=None):
        self.names = names or []
        self.sequence = sequence
        self.target = target
        self.candidate = candidate
        self.bad = bad
        self.requests = []

    def respond(self, request):
        p = json.loads(request.content)
        self.requests.append(p)
        assert p["max_completion_tokens"] == 10000 and p["tool_choice"] == "auto"
        assert AMBIGUOUS not in request.content.decode()
        assert hashlib.sha256(AMBIGUOUS.encode()).hexdigest() not in request.content.decode()
        assert {t["function"]["name"] for t in p["tools"]} == {
            "triage_protein",
            "investigate_pet_evidence",
            "advise_next_experiment",
        }
        replies = [json.loads(m["content"]) for m in p["messages"] if m["role"] == "tool"]
        if not replies:
            args = [{"name": name} for name in self.names] + (
                [{"sequence_input_ref": "turn:sequence"}] if self.sequence else []
            )
            calls = [
                {
                    "id": f"i{i}",
                    "type": "function",
                    "function": {"name": "triage_protein", "arguments": json.dumps(a)},
                }
                for i, a in enumerate(args)
            ]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
            finish = "tool_calls"
        else:
            ids = replies[-1]["identity_required_fact_ids"]
            kind = "CONFIRM_CANDIDATE" if self.candidate else "IDENTIFIER"
            question = (
                f"Did you mean {self.candidate} for {self.target}?"
                if self.candidate
                else f"Could you provide an accession, DOI or FASTA sequence for {self.target}?"
            )
            content = {
                "action": "ASK_CLARIFICATION",
                "target": self.target,
                "request": kind,
                "candidate": self.candidate,
                "question": question,
                "fact_ids": ids,
            }
            if self.bad == "claim":
                content["question"] += " This proves superior PET activity."
            if self.bad == "ids":
                content["fact_ids"] = []
            if self.bad == "candidate":
                content.update(
                    request="CONFIRM_CANDIDATE",
                    candidate="LCC-ICCG",
                    question=f"Did you mean LCC-ICCG for {self.target}?",
                )
            message = {"role": "assistant", "content": json.dumps(content)}
            finish = "length" if self.bad == "length" else "stop"
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "completion_tokens_details": {"reasoning_tokens": 30},
                },
            },
        )


def run_turn(payload, scenario):
    question, context, sequence = prepare_dialogue_turn(payload)
    backend = MetaModelClient(
        transport=httpx.MockTransport(scenario.respond),
        completion_tokens=10000,
        max_estimated_cost_usd=1,
    )
    return investigate_with_backend(
        question, backend, dialogue_context=context, local_sequence=sequence
    )


def test_three_turn_identity_rejection_sequence_and_cumulative_budgets():
    first = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"},
        DialogueTransport(["ESM050", "LCCMC"]),
    )["result"]
    assert first["dialogue"]["outcome"] == "ASK_CLARIFICATION"
    context = first["dialogue"]["context"]
    assert context["confirmed_labels"] == ["ESM050", None] and context["model_attempts"] == 2
    second = run_turn(
        {"context": context, "text": "No, I do not mean LCC-ICCG.", "input_kind": "ANSWER"},
        DialogueTransport(["LCCMC"]),
    )["result"]
    context = second["dialogue"]["context"]
    assert context["rejected_labels"][1] == ["LCC-ICCG"] and context["model_attempts"] == 4
    third = run_turn(
        {"context": context, "text": "Here is the sequence:\n" + AMBIGUOUS, "input_kind": "ANSWER"},
        DialogueTransport(sequence=True),
    )["result"]
    assert third["dialogue"]["outcome"] == "ASK_CLARIFICATION"
    assert (
        third["workflow_status"] == "PARTIAL"
        and third["synthesis"]["status"] == "VALIDATED_CLARIFICATION"
    )
    context = third["dialogue"]["context"]
    assert context["model_attempts"] == 6 and context["tool_attempts"] == 4
    assert context["temperatures_c"] == [40, 60] and context["maximum_ph"] == 7
    assert (
        context["requested_labels"] == ["ESM050", "LCCMC"]
        and context["confirmed_labels"][1] is None
    )
    text = json.dumps(third)
    assert AMBIGUOUS not in text and "AMBIGUOUS_RESIDUES" in text and '"J"' in text
    assert validate_dialogue_context(context)
    fourth = run_turn(
        {"context": context, "text": "No", "input_kind": "ANSWER"}, DialogueTransport(["LCCMC"])
    )
    assert fourth["result"]["dialogue"]["context"]["model_attempts"] == 8
    assert fourth["result"]["dialogue"]["context"]["usage"]["actual_http_requests"] == 8
    from zymecycle.runtime_config import RuntimeLimits

    exhausted = fourth["result"]["dialogue"]["context"]
    exhausted["model_attempts"] = RuntimeLimits().session_models
    blocked = run_turn(
        {"context": exhausted, "text": "No", "input_kind": "ANSWER"}, DialogueTransport(["LCCMC"])
    )
    assert blocked["status"] == "ERROR" and blocked["result"] is None


@pytest.mark.parametrize("bad", ["claim", "ids", "candidate", "length"])
def test_unverified_questions_claims_and_incomplete_responses_are_withheld(bad):
    scenario = DialogueTransport(["ESM050", "LCCMC"], bad=bad)
    result = run_turn({"context": None, "text": QUESTION, "input_kind": "ANSWER"}, scenario)
    if bad == "claim":
        assert result["result"]["dialogue"]["outcome"] == "ASK_CLARIFICATION"
        assert (
            result["result"]["dialogue"]["clarification"]["presentation"]
            == "DETERMINISTIC_FALLBACK"
        )
    else:
        assert result["result"]["dialogue"]["outcome"] == "INCOMPLETE"
        assert result["result"]["synthesis"]["status"] == "INCOMPLETE"
    assert len(scenario.requests) == 2
    assert "This proves superior PET activity" not in json.dumps(result)


def test_generalized_source_backed_spelling_confirmation():
    question = QUESTION.replace("ESM050", "TEP021").replace("LCCMC", "DP 038")
    result = run_turn(
        {"context": None, "text": question, "input_kind": "ANSWER"},
        DialogueTransport(["TEP021", "DP 038"], target="DP 038", candidate="DP038"),
    )
    c = result["result"]["dialogue"]["context"]
    assert c["confirmed_labels"] == ["TEP021", None] and c["pending_candidate"] == "DP038"
    _, rejected, _ = prepare_dialogue_turn({"context": c, "text": "No", "input_kind": "ANSWER"})
    assert rejected["rejected_labels"][1] == ["DP038"]
    result = run_turn(
        {"context": rejected, "text": "No", "input_kind": "ANSWER"},
        DialogueTransport(["DP 038"], target="DP 038", candidate="DP038"),
    )
    assert result["result"]["dialogue"]["outcome"] == "INCOMPLETE"


def test_raw_fasta_private_boundary_and_exact_reference_does_not_establish_function():
    from zymecycle.observations import load_nrel, select_nrel_protein

    initial = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"},
        DialogueTransport(["ESM050", "LCCMC"]),
    )["result"]["dialogue"]["context"]
    seq = select_nrel_protein(load_nrel()["result"], "DP038")["result"]["amino_acid_sequence"]

    class Exact(DialogueTransport):
        def respond(self, request):
            assert (
                seq not in request.content.decode()
                and "PRIVATE_HEADER" not in request.content.decode()
            )
            response = super().respond(request)
            data = response.json()
            if data["choices"][0]["finish_reason"] == "stop":
                c = json.loads(data["choices"][0]["message"]["content"])
                c.update(
                    request="CONDITIONS",
                    question="Could you provide the exact pH and PET form for LCCMC?",
                )
                data["choices"][0]["message"]["content"] = json.dumps(c)
            return httpx.Response(200, json=data)

    result = run_turn(
        {"context": initial, "text": ">PRIVATE_HEADER\n" + seq, "input_kind": "SEQUENCE"},
        Exact(sequence=True),
    )
    view = result["result"]
    assert view["dialogue"]["outcome"] == "ASK_CLARIFICATION"
    assert view["dialogue"]["context"]["confirmed_labels"] == ["ESM050", "DP038"]
    assert seq not in json.dumps(result) and "PRIVATE_HEADER" not in json.dumps(result)
    assert "experimental construct identity" in json.dumps(result)


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_ph": 99},
        {"model_attempts": -1},
        {"confirmed_labels": ["ESM050", "LCCMC"]},
        {"extra": "private"},
    ],
)
def test_forged_context_is_rejected_before_model(change):
    _, context, _ = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )
    context.update(change)
    assert not validate_dialogue_context(context)


def test_unsafe_sequence_model_echo_is_blocked_before_scientific_execution():
    from zymecycle.agent import run_investigation

    _, context, seq = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )

    class Echo:
        mode = "MOCK_META"

        def complete(self, *args):
            return {"role": "assistant", "content": AMBIGUOUS}

    r = run_investigation(
        "Safe identity clarification", Echo(), dialogue_context=context, local_sequence=AMBIGUOUS
    )
    assert r["result"]["tool_attempts"] == 0 and AMBIGUOUS not in json.dumps(r)


def test_scientific_scope_guard_rejects_unresolved_names_invented_ph_and_ranking():
    from zymecycle.agent_types import AgentState
    from zymecycle.dialogue import guard_dialogue_tool

    _, context, _ = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )
    context["confirmed_labels"][0] = "ESM050"
    state = AgentState("data/nrel/label_data.csv", dialogue_context=context)
    guard_dialogue_tool(
        "investigate_pet_evidence",
        {"query": {"protein_name": "ESM050", "temperature_c": 40}},
        state,
    )
    for name, args in [
        ("investigate_pet_evidence", {"query": {"protein_name": "LCCMC"}}),
        ("investigate_pet_evidence", {"query": {"protein_name": "ESM050", "ph": 7}}),
        ("investigate_pet_evidence", {"query": {"protein_name": "ESM050", "temperature_c": 80}}),
        ("advise_next_experiment", {"enzyme_identifier": "ESM050"}),
    ]:
        with pytest.raises(ValueError):
            guard_dialogue_tool(name, args, state)


def test_no_source_data_writes_or_logging_during_private_sequence_turn(monkeypatch, capsys, caplog):
    import builtins
    import io

    first = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"},
        DialogueTransport(["ESM050", "LCCMC"]),
    )["result"]["dialogue"]["context"]
    original = builtins.open
    original_io = io.open

    def readonly(opener):
        def wrapped(path, mode="r", *args, **kwargs):
            assert not set(mode) & set("wax+")
            return opener(path, mode, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(builtins, "open", readonly(original))
    monkeypatch.setattr(io, "open", readonly(original_io))
    result = run_turn(
        {"context": first, "text": AMBIGUOUS, "input_kind": "SEQUENCE"},
        DialogueTransport(sequence=True),
    )
    assert result["result"]["dialogue"]["outcome"] == "ASK_CLARIFICATION"
    captured = capsys.readouterr()
    assert AMBIGUOUS not in captured.out + captured.err + caplog.text


def test_existing_scientific_recommendation_contract_remains_canonical():
    from test_research_advisor_integration import AdviceTransport

    from zymecycle.agent import run_investigation

    scenario = AdviceTransport()
    q = "What experiment should I perform next to investigate ESM050 at 40°C versus 60°C, at pH 6.5, using crystalline PET powder? Use NREL and PAZy evidence."
    r = run_investigation(
        q, MetaModelClient(transport=httpx.MockTransport(scenario.respond), completion_tokens=10000)
    )["result"]
    assert r["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert r["validation"]["selection_counts"] == {"required": 23, "selected": 23, "omitted": 0}
    assert r["status"] == "PARTIAL"
    assert any("0.08499999999999999" in f["text"] for f in r["facts"])
    assert any("0.13" in f["text"] for f in r["facts"])


def test_plain_c_temperature_scope_matches_degree_notation_without_identifier_digits():
    from zymecycle.agent_request import interpret_request

    a = interpret_request(
        "What experiment should I perform next to investigate ESM050 at 40C versus 60C, at pH 6.5 using crystalline PET powder?"
    )
    b = interpret_request(
        "What experiment should I perform next to investigate ESM050 at 40°C versus 60°C, at pH 6.5 using crystalline PET powder?"
    )
    assert a["temperatures"] == b["temperatures"] == [40, 60]
    assert not interpret_request("Investigate DP040C identity")["temperatures"]


def test_real_advisor_continuation_after_explicit_within_bound_conditions_and_focus():
    from zymecycle.observations import load_nrel, select_nrel_protein

    first = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"},
        DialogueTransport(["ESM050", "LCCMC"]),
    )["result"]["dialogue"]["context"]
    # Exact local match establishes a reference label; it is not an assayed construct proof.
    sequence = select_nrel_protein(load_nrel()["result"], "DP038")["result"]["amino_acid_sequence"]
    context = deepcopy(first)
    context["confirmed_labels"][1] = "DP038"
    context["pending_index"] = None
    payload = {
        "context": context,
        "text": "Please investigate ESM050 at pH 6.5, crystalline PET powder",
        "input_kind": "ANSWER",
    }
    question, c, private = prepare_dialogue_turn(payload)
    assert (
        private is None
        and c["assay_ph"] == 6.5
        and c["maximum_ph"] == 7
        and c["focus_label"] == "ESM050"
    )
    assert c["requested_labels"] == ["ESM050", "LCCMC"] and c["temperatures_c"] == [40, 60]
    requests = []

    def respond(request):
        p = json.loads(request.content)
        requests.append(p)
        assert sequence not in request.content.decode()
        replies = [json.loads(m["content"]) for m in p["messages"] if m["role"] == "tool"]
        if not replies:
            a = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
            b = {**a, "temperature_c": 60}
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "advice",
                        "type": "function",
                        "function": {
                            "name": "advise_next_experiment",
                            "arguments": json.dumps(
                                {"enzyme_identifier": "ESM050", "condition_a": a, "condition_b": b}
                            ),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {"fact_ids": replies[-1]["final_selection"]["required_fact_ids"]}
                ),
            }
            finish = "stop"
        return httpx.Response(
            200, json={"model": MODEL, "choices": [{"message": message, "finish_reason": finish}]}
        )

    view = investigate_with_backend(
        question,
        MetaModelClient(transport=httpx.MockTransport(respond), completion_tokens=10000),
        dialogue_context=c,
    )["result"]
    assert view["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert view["workflow_status"] == "PARTIAL"  # Not a validated two-enzyme choice.
    assert view["selection_counts"] == {"required": 23, "selected": 23, "omitted": 0}
    assert view["dialogue"]["context"]["requested_labels"] == ["ESM050", "LCCMC"]
    assert {o["original_text"] for o in view["observations"]} == {"0.08499999999999999", "0.13"}
    assert len(requests) == 2
    for text in (
        "pH 8, crystalline PET powder",
        "Please investigate LCC-ICCG at pH 6.5, crystalline PET powder",
    ):
        with pytest.raises(ValueError):
            prepare_dialogue_turn({"context": context, "text": text, "input_kind": "ANSWER"})


def test_scientific_dialogue_cannot_omit_required_source_facts_even_when_partial():
    from zymecycle.agent import run_investigation

    _, c, _ = prepare_dialogue_turn({"context": None, "text": QUESTION, "input_kind": "ANSWER"})
    c["confirmed_labels"] = ["ESM050", None]
    c["pending_index"] = 1

    class Missing:
        mode = "MOCK_META"

        def complete(self, messages, tools):
            replies = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            if not replies:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "source",
                            "type": "function",
                            "function": {
                                "name": "investigate_pet_evidence",
                                "arguments": json.dumps(
                                    {
                                        "query": {"protein_name": "ESM050", "temperature_c": 40},
                                        "limit": 1,
                                    }
                                ),
                            },
                        }
                    ],
                }
            # Valid canonical warning IDs alone do not satisfy source-evidence completeness.
            ids = [f["id"] for f in replies[-1]["facts"] if f["kind"] == "LIMITATION"]
            return {"role": "assistant", "content": json.dumps({"fact_ids": ids})}

    r = run_investigation("Bounded research decision", Missing(), dialogue_context=c)["result"]
    assert r["synthesis"]["status"] == "INCOMPLETE"
    assert r["validation"]["selection_counts"]["omitted"] > 0


def test_explicit_rejection_of_an_established_label_reopens_only_that_identity():
    _, context, _ = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )
    context["confirmed_labels"] = ["ESM050", "DP038"]
    context["pending_index"] = None
    _, updated, _ = prepare_dialogue_turn(
        {"context": context, "text": "No, I do not mean DP038.", "input_kind": "ANSWER"}
    )
    assert updated["confirmed_labels"] == ["ESM050", None]
    assert updated["rejected_labels"][1] == ["DP038"] and updated["pending_index"] == 1
    assert updated["requested_labels"] == ["ESM050", "LCCMC"]
    assert updated["maximum_ph"] == 7 and updated["temperatures_c"] == [40, 60]


def test_plain_information_request_does_not_claim_lookup_failure_or_require_fictitious_target_evidence():
    # Original live failure is preserved in BUILD_11_REPORT.md. This final question is synthetic.
    result = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}, DialogueTransport(["ESM050"])
    )
    view = result["result"]
    assert view["dialogue"]["outcome"] == "ASK_CLARIFICATION"
    assert view["dialogue"]["clarification"]["presentation"] == "MODEL_VALIDATED"
    assert view["dialogue"]["context"]["confirmed_labels"] == ["ESM050", None]
    assert view["dialogue"]["context"]["temperatures_c"] == [40, 60]
    assert view["dialogue"]["context"]["maximum_ph"] == 7
    assert view["selection_counts"] == {"required": 1, "selected": 1, "omitted": 0}
    assert len(view["progress"]) == 1 and view["progress"][0]["data"]["model_arguments"] == {
        "name": "ESM050"
    }
    assert "not found" not in view["dialogue"]["clarification"]["question"]


def test_safe_diagnostics_record_actual_omitted_count_without_model_prose():
    scenario = DialogueTransport(["ESM050"], bad="ids")
    view = run_turn({"context": None, "text": QUESTION, "input_kind": "ANSWER"}, scenario)["result"]
    diagnostics = view["dialogue"]["diagnostics"]
    assert diagnostics == {
        "failure_code": "OMITTED_SOURCE_FACTS",
        "submitted_ids": 0,
        "required_ids": 1,
        "omitted_ids": 1,
        "action": "ASK_CLARIFICATION",
    }
    assert "question" not in diagnostics


def test_plain_natural_clarification_with_no_factual_claims_needs_no_invented_lookup():
    from zymecycle.agent import run_investigation

    _, context, _ = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )

    class Ask:
        mode = "MOCK_META"

        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action": "ASK_CLARIFICATION",
                        "target": "LCCMC",
                        "request": "IDENTIFIER",
                        "candidate": None,
                        "question": "For LCCMC, what accession or publication should I look at?",
                        "fact_ids": [],
                        "preserved_constraints": {"temperatures_c": [60, 40], "maximum_ph": 7},
                    }
                ),
            }

    report = run_investigation("Research choice context", Ask(), dialogue_context=context)["result"]
    assert report["dialogue"]["clarification"]["presentation"] == "MODEL_VALIDATED"
    assert report["dialogue"]["context"]["pending_index"] == 1
    assert report["dialogue"]["context"]["scientific_status"] == "PARTIAL"
    assert report["validation"]["selection_counts"] == {"required": 0, "selected": 0, "omitted": 0}
    assert report["model_requests"] == 1 and report["tool_attempts"] == 0


def test_supplied_sequence_cannot_be_described_or_clarified_without_real_local_triage():
    from zymecycle.agent import run_investigation

    _, context, _ = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )

    class Skip:
        mode = "MOCK_META"

        def complete(self, *args):
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action": "ASK_CLARIFICATION",
                        "target": "LCCMC",
                        "request": "IDENTIFIER",
                        "candidate": None,
                        "question": "Could you provide a complete sequence for LCCMC?",
                        "fact_ids": [],
                    }
                ),
            }

    report = run_investigation(
        "Private local input supplied", Skip(), dialogue_context=context, local_sequence=AMBIGUOUS
    )["result"]
    assert report["synthesis"]["status"] == "INCOMPLETE"
    assert report["dialogue"]["diagnostics"]["failure_code"] == "LOCAL_SEQUENCE_NOT_TRIAGED"
    assert AMBIGUOUS not in json.dumps(report)


def test_session_limit_is_distinct_from_per_turn_and_exhaustion_sends_no_http():
    from zymecycle.runtime_config import RuntimeLimits

    limits = RuntimeLimits(turn_models=2, turn_tools=2, session_models=4, session_tools=4)
    first = DialogueTransport(["ESM050", "LCCMC"])
    q, c, seq = prepare_dialogue_turn({"context": None, "text": QUESTION, "input_kind": "ANSWER"})

    def call(question, context, scenario):
        return investigate_with_backend(
            question,
            MetaModelClient(
                transport=httpx.MockTransport(scenario.respond), completion_tokens=10000
            ),
            dialogue_context=context,
            runtime_limits=limits,
        )

    one = call(q, c, first)["result"]
    q, c, _ = prepare_dialogue_turn(
        {
            "context": one["dialogue"]["context"],
            "text": "No, I don't mean LCC-ICCG.",
            "input_kind": "ANSWER",
        }
    )
    two = call(q, c, DialogueTransport(["LCCMC"]))["result"]
    assert two["dialogue"]["context"]["usage"]["actual_http_requests"] == 4
    assert two["dialogue"]["context"]["usage"]["prompt_tokens"] == 400
    assert two["dialogue"]["context"]["usage"]["completion_tokens"] == 200
    assert two["dialogue"]["context"]["usage"]["reasoning_tokens"] == 120
    assert two["dialogue"]["context"]["usage"]["reserved_cost_usd"] > 0
    blocked = call(q, two["dialogue"]["context"], DialogueTransport(["LCCMC"]))
    assert blocked["status"] == "ERROR" and blocked["result"] is None
    _, fresh, _ = prepare_dialogue_turn({"context": None, "text": QUESTION, "input_kind": "ANSWER"})
    assert fresh["usage"]["actual_http_requests"] == 0 and fresh["model_attempts"] == 0


def test_nondefault_turn_budget_is_explicit_and_reported():
    from zymecycle.agent import run_investigation
    from zymecycle.runtime_config import RuntimeLimits

    class Stop:
        mode = "MOCK_META"

        def complete(self, *args):
            return {"role": "assistant", "content": "{}"}

    limits = RuntimeLimits(turn_models=8, turn_tools=12, session_models=40, session_tools=60)
    report = run_investigation("Evidence for ESM050", Stop(), runtime_limits=limits)["result"]
    assert report["limits"]["turn_models"] == 8 and report["limits"]["turn_tools"] == 12


def test_explicit_rejection_routes_to_unresolved_identity_when_pending_question_is_conditions():
    _, context, _ = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )
    context["confirmed_labels"] = ["ESM050", None]
    context["pending_index"] = 0
    context["pending_intent"] = "CONDITIONS"
    context["pending_question"] = "What exact pH and PET form should be investigated for ESM050?"
    _, updated, _ = prepare_dialogue_turn(
        {"context": context, "text": "No, I don't mean LCC-ICCG.", "input_kind": "ANSWER"}
    )
    assert updated["rejected_labels"] == [[], ["LCC-ICCG"]]
    assert updated["confirmed_labels"] == ["ESM050", None]
    assert updated["temperatures_c"] == [40, 60] and updated["maximum_ph"] == 7
    assert updated["assay_ph"] is None and updated["requested_labels"] == ["ESM050", "LCCMC"]
    assert context["rejected_labels"] == [[], []]


def test_explicit_rejection_does_not_guess_between_two_unresolved_identities():
    _, context, _ = prepare_dialogue_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}
    )
    with pytest.raises(ValueError, match="Rejection target is ambiguous"):
        prepare_dialogue_turn(
            {"context": context, "text": "No, I don't mean LCC-ICCG.", "input_kind": "ANSWER"}
        )
    assert context["rejected_labels"] == [[], []]


@pytest.mark.parametrize(
    "wording",
    [
        "Before comparing ESM050 and LCCMC, could you clarify which protein you mean by LCCMC?",
        "Could you share the full-length FASTA sequence or a publication DOI for LCCMC?",
    ],
)
def test_natural_wording_passes_actual_parser_grounding_projection_and_ui_view(wording):
    class Natural(DialogueTransport):
        def respond(self, request):
            response = super().respond(request)
            data = response.json()
            if data["choices"][0]["finish_reason"] == "stop":
                content = json.loads(data["choices"][0]["message"]["content"])
                content["question"] = wording
                data["choices"][0]["message"]["content"] = json.dumps(content)
            return httpx.Response(200, json=data)

    view = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"}, Natural(["ESM050", "LCCMC"])
    )["result"]
    assert view["dialogue"]["clarification"]["question"] == wording
    assert view["dialogue"]["clarification"]["presentation"] == "MODEL_VALIDATED"
    assert view["selection_counts"] == {"required": 2, "selected": 2, "omitted": 0}
    assert view["workflow_status"] == "PARTIAL"
    c = view["dialogue"]["context"]
    assert c["confirmed_labels"] == ["ESM050", None]
    assert c["temperatures_c"] == [40, 60] and c["maximum_ph"] == 7 and c["assay_ph"] is None


def test_unresolved_identity_receives_attention_before_conditions_without_scripting_tools():
    class PrematureConditions(DialogueTransport):
        def respond(self, request):
            response = super().respond(request)
            data = response.json()
            if data["choices"][0]["finish_reason"] == "stop":
                content = json.loads(data["choices"][0]["message"]["content"])
                content.update(
                    target="ESM050",
                    request="CONDITIONS",
                    question="What exact pH and PET form should be investigated for ESM050?",
                )
                data["choices"][0]["message"]["content"] = json.dumps(content)
            return httpx.Response(200, json=data)

    view = run_turn(
        {"context": None, "text": QUESTION, "input_kind": "ANSWER"},
        PrematureConditions(["ESM050", "LCCMC"]),
    )["result"]
    assert view["dialogue"]["outcome"] == "INCOMPLETE"
    assert view["synthesis"]["reason_code"] == "UNRESOLVED_IDENTITY_REQUIRES_ATTENTION"
    assert view["dialogue"]["context"]["confirmed_labels"] == ["ESM050", None]
    assert view["dialogue"]["clarification"] is None
