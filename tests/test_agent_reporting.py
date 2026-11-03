import json
from test_agent_execution import setup, call
from scripts.demo_build03 import QUESTIONS
from petzyme.agent_execution import execute_tool_call
from petzyme.agent_request import interpret_request
from petzyme.agent_reporting import validate_investigation


def executed(setup, *, wrong_condition=False, subset=False):
    state, catalog = setup
    execute_tool_call(call('load', 'load_nrel'), state, catalog)
    execute_tool_call(call('parse', 'parse_nrel', table_ref='load:table'), state, catalog)
    condition = {'temperature_c': 60 if wrong_condition else 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
    filters = {**condition, **({'enzyme_identifier': 'E1'} if subset else {})}
    execute_tool_call(call('filter', 'filter_observations', observations_ref='parse:observations', **filters), state, catalog)
    execute_tool_call(call('summary', 'summarize_condition', observations_ref='filter:observations', **condition), state, catalog)
    execute_tool_call(call('evidence', 'assess_evidence', observations_ref='filter:observations'), state, catalog)
    return state


def test_grounding_and_mandatory_caveats(setup):
    state = executed(setup)
    evidence_ids = [fact['id'] for fact in state.facts if fact['kind'] == 'EVIDENCE']
    validation = validate_investigation(interpret_request(QUESTIONS['A']), state, {'fact_ids': evidence_ids})
    assert validation['grounded'] and validation['complete']
    assert any(fact['kind'] == 'LIMITATION' and fact['id'] in validation['selected_fact_ids'] for fact in state.facts)


def test_reject_freeform_overclaim_and_fabricated_refs(setup):
    state = executed(setup)
    for selection in ({'conclusion': 'E1 performs better at 999 U/mL'}, {'fact_ids': ['fabricated']},
                      {'fact_ids': [state.facts[0]['id']], 'prediction': 999}):
        result = validate_investigation(interpret_request(QUESTIONS['A']), state, selection)
        assert not result['grounded'] and not result['complete']
        assert '999' not in json.dumps(result)


def test_scope_validation_prevents_false_no_data_and_subsets(setup):
    for options in ({'wrong_condition': True}, {'subset': True}):
        # Independent state for each branch, reusing the synthetic data file.
        from petzyme.agent_types import AgentState
        original, catalog = setup
        state = executed((AgentState(original.csv_path), catalog), **options)
        selection = {'fact_ids': [fact['id'] for fact in state.facts]}
        result = validate_investigation(interpret_request(QUESTIONS['A']), state, selection)
        assert not result['complete']
        assert any('scope' in issue or 'mismatched' in issue for issue in result['issues'])


def test_ranking_without_specified_ph_cannot_silently_choose_one(setup):
    state = executed(setup)
    question = 'Which NREL enzymes have the highest responses at 40°C in crystalline PET powder?'
    result = validate_investigation(interpret_request(question), state, {'fact_ids': [f['id'] for f in state.facts]})
    assert not result['complete']


def test_prediction_request_is_explicitly_unsupported(setup):
    state = executed(setup)
    question = 'Predict the highest NREL responses at 40°C and pH 6.5 in powder.'
    result = validate_investigation(interpret_request(question), state, {'fact_ids': [f['id'] for f in state.facts]})
    assert not result['complete']
    assert any('unsupported in Build 03' in issue for issue in result['issues'])
