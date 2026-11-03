from copy import deepcopy
import inspect
import json
import pytest
from test_advisor_candidates import investigation
from test_evidence import A, B, observations, documented
from petzyme.advisor import prioritize_experiments, recommend_next_experiments
from petzyme.advisor_candidates import generate_experiment_candidates
from petzyme.advisor_feasibility import assess_experiment_feasibility
from petzyme.conditions import investigate_conditions


def ranked_inputs(inv):
    actions = generate_experiment_candidates(inv)['result']
    return actions, [assess_experiment_feasibility(c)['result'] for c in actions]


def test_topological_order_stable_when_input_shuffled(investigation):
    actions, feasibility = ranked_inputs(investigation)
    result = prioritize_experiments(actions, feasibility)
    assert result == prioritize_experiments(list(reversed(actions)), list(reversed(feasibility)))
    seen = set()
    for i, c in enumerate(result['result'], 1):
        assert set(c['prerequisites']) <= seen
        assert c['priority'] == i and 'ordinal' in c['ranking_rationale']
        seen.add(c['action_id'])
    categories = [c['category'] for c in result['result']]
    assert categories == sorted(categories)


def test_duplicate_candidates_collapse_without_changing_priority(investigation):
    actions, feasibility = ranked_inputs(investigation)
    assert prioritize_experiments(actions + actions[:1], feasibility) == prioritize_experiments(actions, feasibility)


@pytest.mark.parametrize('error', ['unknown_dependency', 'cycle', 'conflict', 'missing_feasibility', 'duplicate_feasibility'])
def test_invalid_priority_graphs_fail(investigation, error):
    actions, feasibility = ranked_inputs(investigation)
    if error == 'unknown_dependency': actions[0]['prerequisites'] = ['unknown']
    if error == 'cycle': actions[0]['prerequisites'] = [actions[0]['action_id']]
    if error == 'conflict': actions.append({**actions[0], 'proposed_action': 'different'})
    if error == 'missing_feasibility': feasibility.pop()
    if error == 'duplicate_feasibility': feasibility.append(feasibility[0])
    assert prioritize_experiments(actions, feasibility)['status'] == 'ERROR'


def test_stated_blockers_change_ready_action_order(investigation):
    actions, feasibility = ranked_inputs(investigation)
    verification = [c for c in actions if c['category'] == 'EVIDENCE_VERIFICATION' and not c['prerequisites']]
    first = min(verification, key=lambda c: c['code'])
    next(f for f in feasibility if f['action_id'] == first['action_id'])['status'] = 'BLOCKED'
    ranked = prioritize_experiments(actions, feasibility)['result']
    assert ranked[0]['action_id'] != first['action_id']
    assert 'BLOCKED' in next(c for c in ranked if c['action_id'] == first['action_id'])['ranking_rationale']


def test_bounded_plan_preserves_dependency_closure_and_omissions(investigation):
    for bound in range(1, 11):
        result = recommend_next_experiments(investigation, bound)
        plan = result['result']
        seen = set()
        for c in plan['recommendations']:
            assert set(c['prerequisites']) <= seen
            seen.add(c['action_id'])
        assert len(seen) <= bound
        assert len(seen) + plan['omitted_count'] == plan['candidate_count']
        assert plan['scientific_status'] == 'PROPOSED_ONLY'
        assert all(c['evidence_category'] == 'HYPOTHETICAL' for c in plan['recommendations'])
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bound', [0, 11, True, 1.5, '3'])
def test_invalid_bounds(investigation, bound):
    assert recommend_next_experiments(investigation, bound)['status'] == 'ERROR'


def test_verified_complete_comparison_does_not_require_new_experiments(observations):
    inv = investigate_conditions(observations, A, B, enzyme_identifier='E1', metadata=documented(observations))['result']
    result = recommend_next_experiments(inv)
    assert result['status'] == 'NO_DATA' and result['result']['recommendations'] == []


def test_proposals_do_not_change_original_evidence(investigation):
    before = deepcopy(investigation)
    plan = recommend_next_experiments(investigation, optional_constraints={'allow_new_measurements': False})['result']
    assert investigation == before
    assert all(c['feasibility']['status'] == 'BLOCKED' for c in plan['recommendations'] if c['category'] == 'PROPOSED_EXPERIMENT')


def test_integration_docstrings_use_flow_contract():
    for f in [generate_experiment_candidates, assess_experiment_feasibility,
              prioritize_experiments, recommend_next_experiments]:
        doc = inspect.getdoc(f)
        assert doc.startswith('Flow: Receives ') and '; returns ToolResult[' in doc


def test_advisor_is_independent_of_io_and_llm(investigation, monkeypatch, caplog, capsys):
    import builtins
    from pathlib import Path
    def blocked(*args, **kwargs):
        raise AssertionError('Advisor must not persist or retrieve scientific records')
    monkeypatch.setattr(builtins, 'open', blocked)
    monkeypatch.setattr(Path, 'open', blocked)
    result = recommend_next_experiments(investigation)
    assert result['status'] == 'SUCCESS'
    assert capsys.readouterr().out == '' and not caplog.records
