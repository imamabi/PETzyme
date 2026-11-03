"""Scientific relevance checks using real local evidence and an unpersuadable mock selector."""

import json

import pytest

from zymecycle.agent import run_investigation
from zymecycle.agent_reporting import format_research_report
from zymecycle.agent_request import interpret_request
from zymecycle.scientific_scope import assess_scientific_scope, interpret_scientific_scope

POLYPROPYLENE = "Compare ESM050 on polypropylene powder at 40C and 60C, pH 6.5."
KINETIC = "Compare kcat/KM of ESM050 on PET powder at 40C and 60C, pH 6.5."
ENDPOINT = "Compare the recorded PET responses for ESM050 on PET powder at 40C and 60C, pH 6.5."


class AllPetFacts:
    mode = "MOCK_META"

    def __init__(self):
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(messages.copy())
        replies = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        a = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
        steps = [
            ("load_nrel", {}),
            ("parse_nrel", {"table_ref": "s0:table"}),
            (
                "investigate_conditions",
                {
                    "observations_ref": "s1:observations",
                    "enzyme_identifier": "ESM050",
                    "condition_a": a,
                    "condition_b": {**a, "temperature_c": 60},
                },
            ),
        ]
        if len(replies) < len(steps):
            name, args = steps[len(replies)]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "s" + str(len(replies)),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": json.dumps({"fact_ids": [f["id"] for r in replies for f in r["facts"]]}),
        }


@pytest.mark.parametrize("question", [POLYPROPYLENE, KINETIC])
def test_all_authentic_pet_ids_cannot_complete_an_incompatible_question(question):
    backend = AllPetFacts()
    out = run_investigation(question, backend)
    report = out["result"]
    assert report["validation"]["grounded"]
    assert not report["validation"]["complete"] and out["status"] == "PARTIAL"
    assert report["validation"]["selection_counts"] == {"required": 3, "selected": 3, "omitted": 0}
    scope = report["validation"]["scientific_scope"]
    assert scope["status"] == "MISMATCH" and not scope["relevance_supported"]
    assert scope["background_only"]
    assert scope["evidence"]["materials"] == ["PET"]
    assert scope["evidence"]["quantities"] == ["RECORDED_RESPONSE"]
    assert report["synthesis"]["status"] == "INCOMPLETE"
    rendered = format_research_report(report)
    assert "background only" in rendered.lower()
    expected = (
        "polypropylene degradation" if question == POLYPROPYLENE else "do not establish kcat/KM"
    )
    assert expected in rendered
    assert any(expected in str(m) for m in backend.requests[-1])
    assert not any("sequence_sha256" in str(m) or "/Users/" in str(m) for m in backend.requests)
    measured = [f["text"] for f in report["facts"] if f["evidence_category"] == "MEASURED"]
    assert any("0.08499999999999999" in t for t in measured)
    assert any("0.13" in t for t in measured)


def test_supported_recorded_pet_response_query_preserves_existing_warnings():
    out = run_investigation(ENDPOINT, AllPetFacts())
    report = out["result"]
    assert out["status"] == "SUCCESS" and report["validation"]["complete"]
    assert report["validation"]["scientific_scope"]["status"] == "MATCH"
    assert report["validation"]["scientific_scope"]["relevance_supported"]
    text = format_research_report(report)
    assert "UNKNOWN/UNVERIFIED" in text and "UNCERTAINTY_UNREPORTED" in text
    assert "direct quantitative comparison supported: False" in text


@pytest.mark.parametrize(
    "material", ["polypropylene", "POLYPROPYLENE", "PP", "polypropene", "polystyrene", "PLA"]
)
def test_explicit_other_material_synonyms(material):
    scope = interpret_scientific_scope(f"Compare ESM050 on {material} powder at 40C and 60C.")
    assert scope["material"] == "OTHER_POLYMER"


@pytest.mark.parametrize(
    "material", ["PET", "pet", "polyethylene terephthalate", "poly(ethylene terephthalate)"]
)
def test_pet_material_synonyms_are_not_polyethylene(material):
    scope = interpret_scientific_scope(f"Compare recorded responses on {material} powder.")
    assert scope["material"] == "PET" and scope["materials"] == ["PET"]


@pytest.mark.parametrize(
    "quantity",
    [
        "kcat/KM",
        "KCAT / Km",
        "k_cat/K_m",
        "kcat over Km",
        "kcat/Kₘ",
        "kcat Km^-1",
        "catalytic efficiency",
        "kinetic parameters",
        "Vmax",
    ],
)
def test_kinetic_notation_is_not_an_endpoint_response(quantity):
    assert (
        interpret_scientific_scope(f"Compare {quantity} on PET powder.")["quantity"]
        == "KINETIC_QUANTITY"
    )


@pytest.mark.parametrize(
    "question",
    [
        "Compare recorded PET responses, not polypropylene activity.",
        "Compare recorded responses on PET powder; discuss polypropylene separately.",
        "Compare recorded PET responses; discuss polypropylene separately.",
        "Compare recorded PET responses; do not infer kcat/KM.",
        "Compare recorded PET responses; I am not asking for kcat/KM.",
    ],
)
def test_background_or_negated_mentions_do_not_reject_valid_pet_questions(question):
    request = interpret_scientific_scope(question)
    assessment = assess_scientific_scope(
        request,
        {"materials": ["PET"], "quantities": ["RECORDED_RESPONSE"], "source_names": ["NREL"]},
    )
    assert not assessment["issues"]


@pytest.mark.parametrize(
    "question",
    [
        "What additional experiments would be needed to estimate kinetic parameters for PET hydrolysis?",
        "Which measurements are needed to estimate kcat/KM on PET?",
    ],
)
def test_kinetic_measurement_requirements_remain_conditional(question):
    request = interpret_scientific_scope(question)
    assert request["purpose"] == "METHOD_REQUIREMENTS"
    assessment = assess_scientific_scope(
        request,
        {"materials": ["PET"], "quantities": ["RECORDED_RESPONSE"], "source_names": ["NREL"]},
    )
    assert not assessment["issues"] and assessment["background_only"]
    explanation = " ".join(assessment["explanations"])
    assert "HYPOTHETICAL" in explanation and "interfacial model" in explanation
    assert "No kinetic constant" in explanation


def test_applicability_question_explains_non_transfer_without_claiming_pp_activity():
    question = "Can PET activity data establish whether ESM050 degrades polypropylene?"
    out = run_investigation(question, AllPetFacts())
    assessment = out["result"]["validation"]["scientific_scope"]
    assert assessment["request"]["purpose"] == "APPLICABILITY"
    assert assessment["background_only"] and not assessment["relevance_supported"]
    assert "does not establish polypropylene degradation" in format_research_report(out["result"])


@pytest.mark.parametrize(
    "question",
    [
        "Compare the recorded responses.",
        "Compare ESM050 on PET and polypropylene.",
        "Compare ESM050 on PET powder and PP powder.",
        "Compare recorded PET responses and kcat/KM.",
        "Compare responses on an unspecified polymer.",
    ],
)
def test_ambiguous_scope_is_never_reported_as_direct_support(question):
    scope = interpret_scientific_scope(question)
    assessment = assess_scientific_scope(
        scope, {"materials": ["PET"], "quantities": ["RECORDED_RESPONSE"], "source_names": ["NREL"]}
    )
    assert not assessment["relevance_supported"]
    assert assessment["explanations"] and assessment["background_only"]


def test_original_request_keeps_scope_even_when_model_facts_are_all_real():
    request = interpret_request(POLYPROPYLENE)
    assert request["scientific_scope"]["material"] == "OTHER_POLYMER"
    assert request["question"] == POLYPROPYLENE


def test_scope_assessment_compares_actual_material_not_just_question_keywords():
    evidence = {
        "materials": ["polypropylene"],
        "quantities": ["RECORDED_RESPONSE"],
        "source_names": ["synthetic scope contract"],
    }
    other = interpret_scientific_scope("Compare recorded responses on polypropylene powder.")
    pet = interpret_scientific_scope("Compare recorded responses on PET powder.")
    assert assess_scientific_scope(other, evidence)["status"] == "MATCH"
    assert assess_scientific_scope(pet, evidence)["status"] == "MISMATCH"
    # Existing production tools do not produce this synthetic other-polymer evidence.


def test_rewritten_request_scope_cannot_override_original_question(monkeypatch):
    import zymecycle.agent as agent

    real = agent.interpret_request

    def rewritten(question):
        request = real(question)
        request["scientific_scope"] = interpret_scientific_scope(ENDPOINT)
        return request

    monkeypatch.setattr(agent, "interpret_request", rewritten)
    report = agent.run_investigation(POLYPROPYLENE, AllPetFacts())["result"]
    assert not report["validation"]["complete"]
    assert report["validation"]["scientific_scope"]["request"]["material"] == "OTHER_POLYMER"


@pytest.mark.parametrize("question", [POLYPROPYLENE, KINETIC])
def test_validator_alone_rejects_authentic_ids_without_scope_warning_injection(question):
    from zymecycle.agent_execution import execute_tool_call
    from zymecycle.agent_reporting import validate_investigation
    from zymecycle.agent_request import select_tool_catalog
    from zymecycle.agent_types import AgentState

    request = interpret_request(question)
    state = AgentState("data/nrel/label_data.csv")
    backend, messages = AllPetFacts(), []
    for _ in range(3):
        message = backend.complete(messages, select_tool_catalog(request))
        event = execute_tool_call(message["tool_calls"][0], state, select_tool_catalog(request))
        messages.append(
            {"role": "tool", "content": json.dumps({"facts": [], "refs": event["refs"]})}
        )
    assert all(e["status"] == "SUCCESS" for e in state.events)
    validation = validate_investigation(
        request, state, {"fact_ids": [f["id"] for f in state.facts]}
    )
    assert validation["grounded"] and validation["selection_counts"]["omitted"] == 0
    assert not validation["complete"]
    assert validation["scientific_scope"]["status"] == "MISMATCH"
    assert not any(".scope" in f["id"] for f in state.facts)


def test_method_question_has_conditional_requirements_in_actual_agent_report():
    question = "What additional experiments would be needed to estimate kinetic parameters for PET hydrolysis?"
    report = run_investigation(question, AllPetFacts())["result"]
    text = format_research_report(report)
    assert "HYPOTHETICAL measurement requirements" in text
    assert "interfacial model" in text and "not fitted parameters" in text
    assert not report["validation"]["scientific_scope"]["issues"]


@pytest.mark.parametrize("question", [POLYPROPYLENE, KINETIC])
def test_dialogue_worker_preflight_exposes_scope_reason_without_keychain_or_process(
    monkeypatch, question
):
    from zymecycle.ui_live import run_keychain_investigation

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid dialogue scope must not launch a child")

    monkeypatch.setattr("zymecycle.ui_live.subprocess.Popen", forbidden)
    result = run_keychain_investigation(
        "Local identity dialogue",
        authorized=True,
        dialogue_input={"context": None, "text": question, "input_kind": "ANSWER"},
    )
    assert result["status"] == "ERROR" and result["result"] is None
    assert "polypropylene" in result["message"] or "kcat/KM" in result["message"]


@pytest.mark.parametrize("question", [POLYPROPYLENE, KINETIC])
def test_model_cannot_hide_scope_warning_by_selecting_only_source_facts(question):
    class OnlyEvidence(AllPetFacts):
        def complete(self, messages, tools):
            reply = super().complete(messages, tools)
            if not reply.get("tool_calls"):
                value = json.loads(reply["content"])
                value["fact_ids"] = [i for i in value["fact_ids"] if ".scope" not in i]
                reply["content"] = json.dumps(value)
            return reply

    report = run_investigation(question, OnlyEvidence())["result"]
    assert report["validation"]["grounded"] and not report["validation"]["complete"]
    assert report["validation"]["selection_counts"]["omitted"] == 0
    assert any(".scope" in f for f in report["validation"]["selected_fact_ids"])


@pytest.mark.parametrize("question", [POLYPROPYLENE, KINETIC])
def test_dialogue_does_not_erase_original_unsupported_scope(question):
    from zymecycle.dialogue import prepare_dialogue_turn
    from zymecycle.scientific_scope import ScientificScopeError

    text = (
        "Should I use ESM050 or LCC-ICCG? My equipment supports 40C and 60C, but I cannot exceed pH 7. "
        + question
    )
    with pytest.raises(ScientificScopeError):
        prepare_dialogue_turn(
            {"context": None, "text": text, "input_kind": "ANSWER", "decision_support": True}
        )


def test_rendered_negative_scope_is_prominent_and_keeps_real_values():
    from streamlit.testing.v1 import AppTest

    from zymecycle.ui_adapter import investigate_with_backend

    view = investigate_with_backend(POLYPROPYLENE, AllPetFacts())["result"]

    def show(payload):
        from streamlit_app import render_findings

        render_findings(payload)

    app = AppTest.from_function(show, args=(view,), default_timeout=30).run()
    assert not app.exception
    assert any("background only" in w.value for w in app.warning)
    assert any("polypropylene degradation" in m.value for m in app.markdown)
    assert view["observations"]
