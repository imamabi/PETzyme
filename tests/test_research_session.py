"""Independent tests for session metadata and input privacy."""

import json
from dataclasses import asdict

import pytest

from petzyme.experimental_window import parse_experimental_window
from petzyme.research_session import append_research_turn, new_research_session


def test_transient_cohort_replay_queries_original_subjects_at_new_temperature():
    from petzyme.pet_evidence import investigate_pet_evidence
    from petzyme.research_session import revalidate_cohort_queries

    queries = revalidate_cohort_queries(
        "What are the five enzymes with the best recorded PET-degradation responses?",
        "Which of those five were tested at 60C?",
    )
    assert len(queries) == 5
    results = [investigate_pet_evidence(item["query"], limit=3, source_scope=item["source_scope"]) for item in queries]
    assert all(result["result"] and result["result"]["pazy"]["records"] for result in results)


def test_explicit_new_comparison_filter_does_not_overwrite_previous_group():
    from petzyme.research_session import revalidate_cohort_queries

    origin = 'What are the five enzymes with the best recorded PET-degradation responses?'
    current = 'Compare those at 60C with the previous group.'
    prior = revalidate_cohort_queries(origin, current, 'Which of those were tested at 40C?', prefer_filter=True)
    assert len(prior) == 5 and all(item['query']['temperature_c'] == 40 for item in prior)


def test_public_turns_are_ordered_and_new_session_resets_history():
    session = new_research_session()
    session = append_research_turn(session, "Investigate ESM050 at 40C.")
    session = append_research_turn(session, "What about 60C?")
    assert [turn.number for turn in session.turns] == [1, 2]
    assert session.turns[0].text == "Investigate ESM050 at 40C."
    fresh = new_research_session()
    assert fresh.session_id != session.session_id and not fresh.turns


def test_research_session_and_context_have_no_fifty_turn_quota():
    from petzyme.agent_request import contextualize_request

    session = new_research_session()
    turns = ['Investigate ESM050 PET evidence.'] + ['What about 40C?'] * 55
    for turn in turns:
        session = append_research_turn(session, turn)
    request = contextualize_request('What about 60C?', turns)
    assert len(session.turns) == 56
    assert request['question'] == 'What about 60C?' and request['enzyme_identifiers'] == ['ESM050']


def test_sequence_is_not_retained_in_session():
    sequence = "ALINPRGRSTTPIATRVQRQRPGLIKJKKM"
    session = append_research_turn(new_research_session(), sequence)
    assert session.turns[0].kind == "private_input" and session.turns[0].text is None
    assert sequence not in json.dumps(asdict(session))


def test_credentials_and_embedded_lowercase_sequences_do_not_enter_context():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    secret = "API key: example-secret-value"
    current, result, _ = dispatch_research_turn(session, secret, run_offline)
    assert current == session and result["status"] == "ERROR"
    sequence = "acdefghiklmnpqrstvwya"
    session = append_research_turn(session, "Please inspect " + sequence)
    assert session.turns[-1].text is None and sequence not in json.dumps(asdict(session))


def test_ph_or_lower_is_a_bound_not_an_exact_setpoint():
    window = parse_experimental_window("What should I test next at pH 7 or lower?")
    assert window["maximum_ph"] == 7
    assert window["ph_values"] == []


def test_named_context_keeps_historical_conditions_separate_from_lab_limits():
    from petzyme.agent_request import contextualize_request

    prior = ["Investigate ESM050 PET evidence.", "My equipment supports only 40C and pH <= 7."]
    historical = contextualize_request("Which were tested at 60C?", prior)
    assert historical["question"] == "Which were tested at 60C?" and historical["temperatures"] == [60]
    planning = contextualize_request("What should I test next?", [*prior, "Which were tested at 60C?"])
    assert planning["window"]["temperatures_c"] == [40] and planning["window"]["maximum_ph"] == 7
    with pytest.raises(ValueError, match="historical evidence or a future experiment"):
        contextualize_request("What about 60C?", [*prior, "What should I test next?"])


def test_real_discovery_and_followups_requery_without_retaining_records():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    for index, question in enumerate((
        "Which PET hydrolases have shown high experimental responses?",
        "Which of those were tested at 40C?",
        "What should I test next at pH 7 or lower?",
    )):
        session, result, replayed = dispatch_research_turn(session, question, run_offline)
        assert replayed and result["result"]
        assert result["result"].get("discovery" if index == 0 else "multi_study")
    assert result["result"]["plan"]
    assert session.discovery_replayable and len(session.turns) == 3
    saved = json.dumps(asdict(session))
    assert "source_protein_id" not in saved and "measurements" not in saved


def test_mismatched_referents_are_not_silently_replaced():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, result, _ = dispatch_research_turn(
        new_research_session(), "Which PET hydrolases have shown high experimental responses?", run_offline
    )
    previous = result["result"]["discovery"]
    session, result, replayed = dispatch_research_turn(session, "Which of those were tested at 60C?", run_offline)
    assert replayed and result["result"]
    current = result["result"]["multi_study"]
    original_ids = {entry["source_protein_id"] for entry in previous["entries"]}
    returned_ids = {entry["protein_id"] for entry in current["pazy"]["records"]}
    assert returned_ids == original_ids and len(returned_ids) == 5
    assert all(entry["conditions"]["temperature_c"] == 60 for entry in current["pazy"]["records"])
    assert not result["result"]["scientific_scope"]["background_only"]


def test_four_turn_historical_query_does_not_inherit_equipment_constraints():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    session, first, _ = dispatch_research_turn(session, "What are the five enzymes with the best recorded PET-degradation responses?", run_offline)
    identifiers = {e["source_protein_id"] for e in first["result"]["discovery"]["entries"]}
    for question in ("Which of those were tested at 40C?", "What should I test next if my equipment supports only 40C and pH <= 7?"):
        session, result, _ = dispatch_research_turn(session, question, run_offline)
        assert result["result"]
    assert session.available_temperatures_c == (40.0,) and session.maximum_ph == 7
    session, result, _ = dispatch_research_turn(session, "Which of those five were tested at 60C?", run_offline)
    records = result["result"]["multi_study"]["pazy"]["records"]
    assert {r["protein_id"] for r in records} == identifiers
    assert all(r["conditions"]["temperature_c"] == 60 for r in records)
    assert session.available_temperatures_c == (40.0,) and session.maximum_ph == 7
    assert result["result"]["tool_attempts"] == 6
    assert all(item.get("actor") == "APPLICATION_COVERAGE" for item in result["result"]["progress"])


def test_cohort_followups_send_only_current_question_to_worker():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, first, _ = dispatch_research_turn(new_research_session(), "What are the five enzymes with the best recorded PET-degradation responses?", run_offline)
    assert first["result"]
    calls = []
    def runner(question, active):
        calls.append(question)
        return run_offline(question, cohort_origin=active.cohort_origin, cohort_filter=active.cohort_filter,
                           cohort_pazy_only=active.cohort_pazy_only, cohort_window=active.cohort_window)
    for question in ("Which of those were tested at 40C?", "What should I test next if my equipment supports only 40C and pH <= 7?", "Which of those five were tested at 60C?"):
        session, result, _ = dispatch_research_turn(session, question, run_offline, context_runner=runner)
        assert result["result"] and calls[-1] == question
    assert "source_protein_id" not in json.dumps(asdict(session))
    session, restricted, _ = dispatch_research_turn(session, "Show PAZy evidence only.", run_offline)
    assert restricted["result"]
    sources = {item["source"]: item["status"] for item in restricted["result"]["source_coverage"]}
    assert sources["NREL"] == "EXCLUDED" and sources["PAZy"] == "FOUND"
    assert all(record["conditions"]["temperature_c"] == 60 for record in restricted["result"]["multi_study"]["pazy"]["records"])
    session, later, _ = dispatch_research_turn(session, "What about pH 7.5?", run_offline)
    assert later["result"] and session.cohort_pazy_only
    assert {item["source"]: item["status"] for item in later["result"]["source_coverage"]}["NREL"] == "EXCLUDED"
    assert session.available_temperatures_c == (40.0,) and session.maximum_ph == 7
    session, both, _ = dispatch_research_turn(session, "Use both sources.", run_offline)
    assert both["result"] and not session.cohort_pazy_only
    assert {item["source"]: item["status"] for item in both["result"]["source_coverage"]}["NREL"] != "EXCLUDED"
    session, limits, _ = dispatch_research_turn(session, "I can only run experiments at 40C now.", run_offline)
    assert limits["status"] == "NO_DATA" and session.available_temperatures_c == (40.0,) and session.maximum_ph == 7
    _, unavailable, _ = dispatch_research_turn(session, "Show NREL evidence only.", run_offline)
    assert unavailable["status"] == "ERROR" and "verified source identities" in unavailable["message"]
    session, next_plan, _ = dispatch_research_turn(session, "What should I test next?", run_offline, context_runner=runner)
    assert next_plan["result"] and next_plan["result"]["plan"]
    for proposal in next_plan["result"]["plan"]["recommendations"]:
        for condition in proposal["target_conditions"]:
            assert condition["temperature_c"] == 40 and condition["ph"] <= 7


def test_self_contained_their_question_is_not_a_missing_cohort_reference():
    from petzyme.ui_adapter import dispatch_research_turn

    question = 'Find PET-associated enzymes in PAZy and show their biological context and supporting publications.'
    seen = []
    def runner(value):
        seen.append(value)
        raise RuntimeError('runner reached')
    with pytest.raises(RuntimeError, match='runner reached'):
        dispatch_research_turn(new_research_session(), question, runner)
    assert seen == [question]


def test_cohort_no_match_and_ambiguous_reference_do_not_fabricate_results():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    session, no_context, _ = dispatch_research_turn(session, "Which of those were tested at 99C?", run_offline)
    assert no_context["status"] == "NO_DATA" and "no previous research group" in no_context["message"]
    assert not session.active_questions
    session, initial, _ = dispatch_research_turn(session, "What are the five enzymes with the best recorded PET-degradation responses?", run_offline)
    assert initial["result"]
    before = session
    session, ambiguous, _ = dispatch_research_turn(session, "Which were tested at 99C?", run_offline)
    assert session == before and ambiguous["status"] == "ERROR" and "Clarify" in ambiguous["message"]
    session, missing, _ = dispatch_research_turn(session, "Which of those were tested at 99C?", run_offline)
    assert missing["result"] and not missing["result"]["multi_study"]["pazy"]["records"]
    assert not missing["result"]["multi_study"]["nrel"]["observations"]
    assert len(missing["result"]["cohort_missing"]) == 5


def test_two_historical_groups_are_replayed_separately_without_reranking():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    for question in (
        'What are the five enzymes with the best recorded PET-degradation responses?',
        'Which of those were tested at 40C?',
        'Which of those were tested at 60C?',
    ):
        session, result, _ = dispatch_research_turn(session, question, run_offline)
        assert result['result'] is not None
    session, result, replayed = dispatch_research_turn(session, 'Compare these results with the previous group.', run_offline)
    assert replayed and result['result'] is not None
    groups = result['result']['comparison_groups']
    assert len(groups) == 2
    assert all(o['temperature_c'] == 40 for o in groups[0]['preview']['nrel']['observations'])
    assert all(o['temperature_c'] == 60 for o in groups[1]['preview']['nrel']['observations'])
    assert all(len(g['preview']['nrel']['observations']) == 5 for g in groups)
    assert 'not established' in result['result']['headline']
    assessments = result['result']['group_assessments']
    assert len(assessments) == 5
    assert all(item['category'] == 'LIMITED_COMPARISON' and item['varying_variable'] == 'temperature_c'
               for item in assessments)
    assert all('verified_units' in item['missing_fields'] for item in assessments)
    assert all(not item.get('numerical_comparison') for item in assessments)
    assert result['result']['selection_counts']['omitted'] == 0
    assert 'source_protein_id' not in json.dumps(asdict(session))


def test_group_comparison_does_not_replace_earlier_source_scope():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    for question in (
        'What are the five enzymes with the best recorded PET-degradation responses?',
        'Which of those were tested at 40C?',
        'Show PAZy evidence only.',
        'Which of those were tested at 60C?',
    ):
        session, result, _ = dispatch_research_turn(session, question, run_offline)
        assert result['result'] is not None
    session, result, replayed = dispatch_research_turn(session, 'Compare these results with the previous group.', run_offline)
    assert replayed and result['result'] is not None
    groups = result['result']['comparison_groups']
    assert len(groups) == 2 and len(groups[0]['preview']['nrel']['observations']) == 5
    assert groups[1]['preview']['nrel']['observations'] == []
    assert 'Source restrictions differ' in result['result']['headline']
    assert session.cohort_compare_scopes is None


def test_ambiguous_group_comparison_does_not_substitute_a_ranking():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, _, _ = dispatch_research_turn(new_research_session(), "What are the five enzymes with the best recorded PET-degradation responses?", run_offline)
    session, result, _ = dispatch_research_turn(session, "Which of those five were tested at 60C?", run_offline)
    assert result["result"]
    _, comparison, _ = dispatch_research_turn(session, "Compare these results with the previous group.", run_offline)
    assert comparison["status"] == "ERROR" and comparison["result"] is None
    assert "comparability" in comparison["message"]


def test_cohort_condition_after_planning_requires_historical_or_future_intent():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, _, _ = dispatch_research_turn(new_research_session(), "What are the five enzymes with the best recorded PET-degradation responses?", run_offline)
    session, plan, _ = dispatch_research_turn(session, "What should I test next at 40C and pH 7 or lower?", run_offline)
    assert plan["result"]
    unchanged, ambiguous, _ = dispatch_research_turn(session, "What about 60C?", run_offline)
    assert unchanged == session and ambiguous["status"] == "NO_DATA"
    assert "historical evidence or a future experiment" in ambiguous["message"]


def test_cohort_provider_payload_is_bounded_and_excludes_private_fields():
    from petzyme.agent import run_investigation

    class Backend:
        mode = "MOCK_META"

        def complete(self, messages, tools):
            assert all(tool["function"]["name"] not in ("advise_next_experiment", "recommend_next_experiments") for tool in tools)
            payload = next(json.loads(item["content"]) for item in messages
                           if item["role"] == "user" and item["content"].startswith('{"untrusted_data": true, "cohort_evidence"'))
            serialized = json.dumps(payload)
            for forbidden in ('"protein_id"', '"sequence"', '"amino_acid_sequence"', '"embedding"', '"rows"', '/Users/'):
                assert forbidden not in serialized
            assert len(payload["cohort_labels"]) == 5
            return {"role": "assistant", "content": json.dumps({"fact_ids": payload["required_fact_ids"]})}

    result = run_investigation("Which of those five were tested at 60C?", Backend(),
                               cohort_origin="What are the five enzymes with the best recorded PET-degradation responses?")
    assert result["result"] and result["result"]["validation"]["grounded"]
    assert result["result"]["tool_attempts"] == 6


def test_group_comparison_provider_payload_preserves_privacy_and_attempt_counts():
    from petzyme.agent import run_investigation

    class Backend:
        mode = 'MOCK_META'

        def complete(self, messages, tools):
            payload = next(json.loads(item['content']) for item in messages
                           if item['role'] == 'user' and item['content'].startswith('{"untrusted_data": true, "cohort_evidence"'))
            assert {item['group'] for item in payload['cohort_evidence']} == {0, 1}
            serialized = json.dumps(payload)
            assert all(field not in serialized for field in ('"protein_id"', '"sequence"', '"embedding"', '"rows"', '/Users/'))
            return {'role': 'assistant', 'content': json.dumps({'fact_ids': payload['required_fact_ids']})}

    result = run_investigation('Compare these results with the previous group.', Backend(),
        cohort_origin='What are the five enzymes with the best recorded PET-degradation responses?',
        cohort_compare_filters=('Which of those were tested at 40C?', 'Which of those were tested at 60C?'))
    assert result['result'] and result['result']['validation']['grounded']
    assert result['result']['tool_attempts'] == 12


def test_named_followup_sends_current_question_and_separate_validated_context():
    from petzyme.agent import run_investigation
    from petzyme.ui_adapter import OfflineWorkflow

    original = "Investigate ESM050 PET evidence."
    current = "What about 60C?"
    delegate = OfflineWorkflow(current, context_turns=[original])

    class Backend:
        mode = "MOCK_META"

        def complete(self, messages, tools):
            assert messages[-1]["content"] == current or messages[-1]["role"] == "tool"
            assert original not in json.dumps(messages)
            return delegate.complete(messages, tools)

    result = run_investigation(current, Backend(), context_turns=[original])
    assert result["result"] and result["result"]["tool_attempts"] > 0
    assert result["result"]["validation"]["scientific_scope"]["request"]["material"] == "PET"


@pytest.mark.parametrize("question,issue", [
    ("What about polypropylene at 60C?", "SCIENTIFIC_SCOPE_MATERIAL_MISMATCH"),
    ("Compare kcat/KM at 60C.", "SCIENTIFIC_SCOPE_QUANTITY_MISMATCH"),
])
def test_named_context_does_not_override_explicit_scientific_mismatches(question, issue):
    from petzyme.ui_adapter import run_offline

    result = run_offline(question, context_turns=["Investigate ESM050 PET recorded responses."])
    assert result["result"] and issue in result["result"]["scientific_scope"]["issues"]


def test_unified_app_three_turns_and_reset_without_paid_inference(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from petzyme import ui_live

    def forbidden(*args, **kwargs):
        raise AssertionError("No provider launch was authorized")

    monkeypatch.setattr(ui_live.subprocess, "Popen", forbidden)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception and not app.sidebar.radio
    for question in (
        "Which PET hydrolases have shown high experimental responses?",
        "Which of those were tested at 40C?",
        "What should I test next at pH 7 or lower?",
    ):
        app.text_area(key="unified_question").input(question)
        app.button(key="FormSubmitter:ask_petzyme-Ask PETzyme").click().run()
        assert not app.exception and not app.error
        assert app.table
    assert any("Suggested next step" in item.value for item in app.markdown)
    assert len(app.session_state["research_session"].turns) == 3
    app.button(key="unified_reset").click().run()
    assert not app.exception and not app.session_state["research_session"].turns
    assert app.text_area(key="unified_question").value == ""


def test_unknown_identity_and_sequence_are_handled_without_private_retention():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, result, _ = dispatch_research_turn(new_research_session(), "Investigate LCCMC.", run_offline)
    assert result["status"] == "NO_DATA" and session.pending_label == "LCCMC"
    assert session.pending_candidate is None and "sequence" in result["message"]
    session, result, _ = dispatch_research_turn(session, "No, that is not my enzyme.", run_offline)
    assert result["status"] == "NO_DATA" and session.pending_label == "LCCMC"
    sequence = "ALINPRGRSTTPIATRVQRQRPGLIKJKKM"
    session, result, _ = dispatch_research_turn(session, sequence, run_offline)
    assert result["result"]["sequence_length"] == len(sequence)
    assert "J" in result["result"]["flagged_symbols"]
    assert session.pending_label == "LCCMC" and session.sequence_label is None
    assert sequence not in json.dumps(asdict(session))


def test_new_named_investigation_can_follow_unresolved_private_sequence():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, result, _ = dispatch_research_turn(new_research_session(), "ALINPRGRSTTPIATRVQRQRPGLIKJKKM", run_offline)
    assert result["result"]["identity_status"] == "UNRESOLVED"
    session, result, _ = dispatch_research_turn(session, "Investigate DP043.", run_offline)
    assert result["result"] and session.enzyme_identifiers == ("DP043",)


def test_named_enzyme_continuation_requeries_without_persisting_source_data():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    for q in ("Investigate ESM050.", "What PAZy evidence is available?", "I can only run at 40C."):
        session, result, _ = dispatch_research_turn(session, q, run_offline)
        assert result["result"] and result["result"].get("multi_study")
    assert len(session.turns) == 3
    assert all("measurements" not in str(turn) for turn in session.turns)


def test_source_only_and_new_objective_do_not_silently_inherit_old_limits():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    for q in ("Investigate ESM050.", "Use PAZy only for the evidence.", "I can only run at 40C."):
        session, result, _ = dispatch_research_turn(session, q, run_offline)
        assert result["result"] and not result["message"]
    assert session.sources == ("PAZy",) and session.available_temperatures_c == (40.0,)
    assert {r["source"]: r["status"] for r in result["result"]["source_coverage"]}["NREL"] == "EXCLUDED"
    session, result, _ = dispatch_research_turn(session, "Investigate DP043.", run_offline)
    assert result["result"] and session.enzyme_identifiers == ("DP043",)
    assert not session.available_temperatures_c and session.sources != ("PAZy",)


def test_explicit_source_change_overrides_prior_source_preference():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, first, _ = dispatch_research_turn(
        new_research_session(), "Investigate ESM050 using NREL only.", run_offline
    )
    assert first["result"] and session.sources == ("NREL",)
    session, second, _ = dispatch_research_turn(session, "Use PAZy only for the evidence.", run_offline)
    assert second["result"] and session.sources == ("PAZy",)
    receipts = {r["source"]: r["status"] for r in second["result"]["source_coverage"]}
    assert receipts["NREL"] == "EXCLUDED" and receipts["PAZy"] == "FOUND"
    assert session.turns[0].text == "Investigate ESM050 using NREL only."
    session, both, _ = dispatch_research_turn(session, "Use both sources.", run_offline)
    assert both["result"] and set(session.sources) == {"NREL", "PAZy"}


def test_offline_source_switching_retrieves_grounded_evidence_and_keeps_planning_partial():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    questions = (
        'Investigate ESM050 using NREL and PAZy experimental evidence.',
        'Show NREL measurements only.',
        'Show PAZy measurements only.',
        'Show measurements from both sources.',
        'What should I test next if I can only use 40°C and pH at most 7?',
    )
    views = []
    for question in questions:
        session, result, _ = dispatch_research_turn(session, question, run_offline)
        assert result['result'] is not None
        views.append(result['result'])
    first, nrel, pazy, both, plan = views
    assert first['multi_study']['nrel']['observations'] and first['multi_study']['pazy']['records']
    assert nrel['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
    assert nrel['selection_counts']['omitted'] == 0 and nrel['observations']
    assert all(o['enzyme_identifier'] == 'ESM050' for o in nrel['observations'])
    assert {r['source']: r['status'] for r in nrel['source_coverage']}['PAZy'] == 'EXCLUDED'
    assert pazy['multi_study']['pazy']['records'] and not pazy['multi_study']['nrel']['observations']
    assert {r['source']: r['status'] for r in pazy['source_coverage']}['NREL'] == 'EXCLUDED'
    assert both['multi_study']['nrel']['observations'] and both['multi_study']['pazy']['records']
    assert plan['workflow_status'] == 'PARTIAL' and plan['plan']
    assert not any('REQUIRED_INVESTIGATION_STEP_MISSING' == v['synthesis']['reason_code'] for v in views)
    assert 'matching_parent_records' not in json.dumps(asdict(session))


def test_nrel_single_temperature_followup_does_not_require_two_condition_comparison():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    for question in ("Investigate ESM050 — what's reported, and what should I test next?",
                     'Show PAZy evidence only for ESM050 at 40°C.',
                     'Show NREL evidence only for ESM050 at 60°C.'):
        session, result, _ = dispatch_research_turn(session, question, run_offline)
    view = result['result']
    assert view is not None and view['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
    assert len(view['observations']) == 4 and all(o['temperature_c'] == 60 for o in view['observations'])
    assert view['selection_counts']['omitted'] == 0
    assert all('investigate_conditions' not in issue for issue in view['issues'])


def test_named_followups_do_not_join_prior_user_turns_into_current_question():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, first, _ = dispatch_research_turn(new_research_session(), "Investigate ESM050 PET evidence.", run_offline)
    assert first["result"]
    seen = []

    def runner(question, active):
        seen.append(question)
        return run_offline(question, context_turns=list(active.active_questions[:-1]))

    session, result, _ = dispatch_research_turn(session, "What about 60C?", run_offline, context_runner=runner)
    assert result["result"] and seen == ["What about 60C?"]
    assert session.enzyme_identifiers == ("ESM050",) and session.temperatures_c == (60.0,)


def test_named_context_retains_separately_stated_equipment_constraints():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, result, _ = dispatch_research_turn(new_research_session(), "Investigate ESM050 PET evidence.", run_offline)
    assert result["result"]
    session, result, _ = dispatch_research_turn(session, "My maximum pH is 7.", run_offline)
    assert result["result"] and session.maximum_ph == 7 and session.enzyme_identifiers == ("ESM050",)
    session, result, _ = dispatch_research_turn(session, "I can only run at 40C.", run_offline)
    assert result["result"] and session.maximum_ph == 7 and session.available_temperatures_c == (40.0,)


def test_custom_runner_is_never_silently_replaced_for_a_followup():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, first, _ = dispatch_research_turn(new_research_session(), "Investigate ESM050 PET evidence.", run_offline)
    assert first["result"]

    def custom_runner(question):
        raise AssertionError("A context-free runner cannot safely handle the follow-up")

    same, result, _ = dispatch_research_turn(session, "What about 60C?", custom_runner)
    assert same == session and result["status"] == "ERROR" and "context-aware" in result["message"]


def test_two_enzyme_next_step_requests_missing_constraints_then_focus_offline():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session = new_research_session()
    for q in ("Investigate ESM050.", "I can only run at 40C.", "Compare with LCC-ICCG."):
        session, result, _ = dispatch_research_turn(session, q, run_offline)
        assert result["result"]
    session, result, _ = dispatch_research_turn(session, "What should I test next?", run_offline)
    assert result["status"] == "NO_DATA" and session.pending_need == "CONSTRAINTS"
    session, result, _ = dispatch_research_turn(session, "pH 7 or lower", run_offline)
    assert result["status"] == "NO_DATA" and session.pending_need == "FOCUS"
    assert session.maximum_ph == 7 and session.ph_values == ()
    session, result, _ = dispatch_research_turn(session, "ESM050", run_offline)
    assert result["result"] and result["result"]["plan"]
    assert session.enzyme_identifiers == ("ESM050",)


def test_new_objective_can_leave_pending_focus_without_old_constraints():
    from petzyme.ui_adapter import dispatch_research_turn, run_offline

    session, result, _ = dispatch_research_turn(
        new_research_session(),
        "What experiment should I perform for ESM050 and LCC-ICCG? My equipment supports 40C and pH 7 or lower.",
        run_offline,
    )
    assert result["status"] == "NO_DATA" and session.pending_need == "FOCUS"
    session, result, _ = dispatch_research_turn(session, "Investigate DP043.", run_offline)
    assert result["result"] and session.enzyme_identifiers == ("DP043",)
    assert session.maximum_ph is None and session.available_temperatures_c == ()


def test_unified_app_withholds_credentials_without_session_storage():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()
    secret = "API key: example-secret-value"
    app.text_area(key="unified_question").input(secret)
    app.button(key="FormSubmitter:ask_petzyme-Ask PETzyme").click().run()
    assert not app.exception and app.error
    assert secret not in str(app.session_state["research_session"])
    assert app.text_area(key="unified_question").value == ""


def test_local_sequence_question_routes_to_existing_bounded_esm_instrument():
    from petzyme.agent_request import interpret_request
    from petzyme.ui_adapter import local_research_plan

    plan = local_research_plan(interpret_request("Use ESM-2 to find proteins similar to ESM050."))
    assert [step[0] for step in plan] == ["load_nrel", "find_similar_nrel_proteins"]
    assert plan[1][1]["enzyme_identifier"] == "ESM050"
    assert plan[1][1]["max_candidates"] == 8


def test_unified_app_sequence_triage_is_transient():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()
    app.text_area(key="unified_question").input("Investigate LCCMC.")
    app.button(key="FormSubmitter:ask_petzyme-Ask PETzyme").click().run()
    assert not app.exception and app.session_state["research_session"].pending_label == "LCCMC"
    assert any("sequence" in item.value for item in app.info)
    sequence = "ALINPRGRSTTPIATRVQRQRPGLIKJKKM"
    app.text_area(key="unified_question").input(sequence)
    app.button(key="FormSubmitter:ask_petzyme-Ask PETzyme").click().run()
    assert not app.exception and any("UNRESOLVED" in item.value and "J" in item.value for item in app.info)
    assert sequence not in json.dumps(asdict(app.session_state["research_session"]))
    assert app.text_area(key="unified_question").value == ""


def test_unified_app_adaptive_dialogue_uses_retained_context_with_mock_transport(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    from test_decision import QUESTION
    from test_decision_integration import PlanningTransport, run

    calls = []

    def fake(question, **kwargs):
        assert kwargs['automatic_runtime'] is True and kwargs['completion_tokens'] == 50000
        payload = kwargs["dialogue_input"]
        calls.append(payload)
        return run(payload, PlanningTransport(continuation=payload["context"] is not None))

    monkeypatch.setattr("petzyme.ui_live.run_keychain_investigation", fake)
    monkeypatch.delenv("PETZYME_OFFLINE", raising=False)
    monkeypatch.setattr("scripts.launch_meta.keychain_item_present", lambda: True)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=60).run()
    assert not app.checkbox
    app.text_area(key="unified_question").input(QUESTION)
    app.button(key="FormSubmitter:ask_petzyme-Ask PETzyme").click().run(timeout=60)
    assert not app.exception and len(calls) == 1 and calls[0]["context"] is None
    assert app.session_state["research_session"].dialogue_context is not None
    app.text_area(key="unified_question").input("I can only run experiments at 40C now.")
    app.button(key="FormSubmitter:ask_petzyme-Ask PETzyme").click().run(timeout=60)
    assert not app.exception and len(calls) == 2 and calls[1]["context"] is not None
    ctx = app.session_state["research_session"].dialogue_context
    assert ctx["temperatures_c"] == [40] and ctx["maximum_ph"] == 7
    assert ctx["decision"]["previous_alternatives"]


def test_automatic_muse_failure_displays_truthful_local_fallback(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from petzyme.results import outcome

    calls = []
    def failing_live(question, **kwargs):
        calls.append(question)
        return outcome('ERROR', None, [], [], None, 'Provider unavailable')

    monkeypatch.delenv('PETZYME_OFFLINE', raising=False)
    monkeypatch.setattr('scripts.launch_meta.keychain_item_present', lambda: True)
    monkeypatch.setattr('petzyme.ui_live.run_keychain_investigation', failing_live)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'), default_timeout=60).run()
    app.text_area(key='unified_question').input('Investigate ESM050 PET experimental evidence.')
    app.button(key='FormSubmitter:ask_petzyme-Ask PETzyme').click().run(timeout=60)
    assert not app.exception and len(calls) == 1
    assert any('deterministic' in item.value.lower() for item in app.caption)
    app.run(timeout=60)
    assert len(calls) == 1


@pytest.mark.parametrize("text", ["", " " * 2, "x" * 16385])
def test_invalid_turn_is_rejected(text):
    with pytest.raises(ValueError):
        append_research_turn(new_research_session(), text)
