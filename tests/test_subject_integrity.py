"""Research-session subject continuity and evidence-boundary regressions."""
from copy import deepcopy

import pytest

from petzyme import agent_execution, ui_adapter
from petzyme.agent_request import contextualize_request
from petzyme.research_session import new_research_session
from petzyme.results import outcome

FIRST = 'Investigate ESM050 using NREL and PAZy.'
PLANNING = 'What experiment should I consider next if my laboratory supports only 40C and pH 7 or lower?'


def advance(session, question):
    return ui_adapter.dispatch_research_turn(session, question, ui_adapter.run_offline)


def evidence_subjects(view):
    subjects = {row['enzyme_identifier'] for row in view.get('observations', [])}
    multi = view.get('multi_study') or {}
    subjects.update(row['enzyme_identifier'] for row in (multi.get('nrel') or {}).get('observations', []))
    subjects.update(row['protein_name'] for row in (multi.get('pazy') or {}).get('records', []))
    return subjects


@pytest.mark.parametrize('planning', [PLANNING, 'What should I test next if my laboratory supports only 40°C and pH 7 or lower?'])
@pytest.mark.parametrize('source_turns,scope', [
    (('Show NREL evidence only.', 'Show PAZy evidence only.'), 'PAZY_ONLY'),
    (('Show PAZy evidence only.', 'Show NREL evidence only.'), 'NREL'),
])
def test_source_switches_and_planning_preserve_subject_and_tool_arguments(monkeypatch, source_turns, scope, planning):
    original = ui_adapter.local_research_plan
    plans = []
    def capture(request):
        plan = original(request)
        plans.append((deepcopy(request), deepcopy(plan)))
        return plan
    monkeypatch.setattr(ui_adapter, 'local_research_plan', capture)
    invoked = []
    original_invoke = agent_execution.invoke_tool
    def capture_invoke(name, arguments):
        if name in ('filter_observations', 'advise_next_experiment'):
            invoked.append((name, {key: deepcopy(arguments.get(key)) for key in
                                   ('enzyme_identifier', 'source_scope', 'request_window')}))
        return original_invoke(name, arguments)
    monkeypatch.setattr(agent_execution, 'invoke_tool', capture_invoke)
    session = new_research_session()
    questions = (FIRST, *source_turns, planning)
    for question in questions:
        session, result, _ = advance(session, question)
        assert result['result'] is not None
        assert session.enzyme_identifiers == ('ESM050',)
        assert evidence_subjects(result['result']) <= {'ESM050'}
    assert session.available_temperatures_c == (40.0,)
    assert session.maximum_ph == 7.0
    request, plan = plans[-1]
    assert request['enzyme_identifiers'] == ['ESM050']
    assert request['window']['temperatures_c'] == [40.0]
    assert request['window']['maximum_ph'] == 7.0
    if scope == 'PAZY_ONLY':
        assert plan == [('advise_next_experiment', {'enzyme_identifier': 'ESM050', 'source_scope': scope})]
        name, args = invoked[-1]
        assert name == 'advise_next_experiment' and args['enzyme_identifier'] == 'ESM050'
        assert args['source_scope'] == 'PAZY_ONLY'
        assert args['request_window']['temperatures_c'] == [40.0]
        assert args['request_window']['maximum_ph'] == 7.0
        assert not result['result'].get('observations')
        assert result['result']['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
    else:
        assert request['sources'] == ['NREL']
        assert next(args for name, args in plan if name == 'filter_observations')['enzyme_identifier'] == 'ESM050'
        assert invoked[-1][0] == 'filter_observations' and invoked[-1][1]['enzyme_identifier'] == 'ESM050'
        assert not result['result'].get('multi_study')


def test_explicit_new_subject_replaces_previous_subject():
    session, _, _ = advance(new_research_session(), FIRST)
    session, result, _ = advance(session, 'Investigate LCC-ICCG using NREL and PAZy.')
    assert result['result'] is not None
    assert session.enzyme_identifiers == ('LCC-ICCG',)
    assert evidence_subjects(result['result']) <= {'LCC-ICCG'}


@pytest.mark.parametrize('question', [PLANNING, 'What should I test next?'])
def test_subjectless_planning_requires_clarification_after_reset(question):
    session, result, _ = advance(new_research_session(), question)
    assert result['status'] == 'NO_DATA' and result['result'] is None
    assert 'identify the enzyme' in result['message'].lower()
    assert not session.active_questions


def test_missing_evidence_does_not_substitute_a_subject():
    session, _, _ = advance(new_research_session(), FIRST)
    def unavailable(question, context):
        assert context.enzyme_identifiers == ('ESM050',)
        return outcome('NO_DATA', None, [], [], None, 'No matching source evidence.')
    updated, result, _ = ui_adapter.dispatch_research_turn(session, PLANNING, lambda _: None, context_runner=unavailable)
    assert updated == session
    assert result['status'] == 'NO_DATA' and result['result'] is None


@pytest.mark.parametrize('source', ['nrel', 'pazy'])
def test_mismatched_source_record_is_rejected_before_presentation(source):
    session, original, _ = advance(new_research_session(), FIRST)
    result = deepcopy(original)
    if source == 'nrel':
        result['result']['multi_study']['nrel']['observations'][0]['enzyme_identifier'] = 'DP003'
    else:
        result['result']['multi_study']['pazy']['records'][0]['protein_name'] = 'DP003'
    updated, rejected, _ = ui_adapter.dispatch_research_turn(
        session, PLANNING, lambda _: None, context_runner=lambda *_: result)
    assert updated == session
    assert rejected['status'] == 'ERROR' and rejected['result'] is None
    assert 'did not match' in rejected['message']


def test_paraphrased_planning_is_interpreted_with_a_laboratory_window():
    request = contextualize_request(PLANNING, [FIRST, 'Show PAZy evidence only.'])
    assert request['scientific_scope']['purpose'] == 'EXPERIMENT_PLANNING'
    assert request['enzyme_identifiers'] == ['ESM050']
    assert request['window']['maximum_ph'] == 7.0
