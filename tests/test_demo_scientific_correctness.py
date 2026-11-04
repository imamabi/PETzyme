"""Focused scientific checks using transient pinned-source evidence and synthetic gaps."""
import json
from copy import deepcopy
from itertools import product

import pytest

from petzyme.agent_execution import execute_tool_call
from petzyme.agent_request import interpret_request, select_tool_catalog
from petzyme.agent_types import AgentState
from petzyme.evidence_records import reported_response
from petzyme.experimental_coverage import summarize_experimental_coverage
from petzyme.experimental_window import parse_planning_goal
from petzyme.nrel_units import standardized_nrel_values
from petzyme.observations import load_nrel, parse_nrel
from petzyme.pet_evidence import investigate_pet_evidence
from petzyme.pet_presenter import preview_pet_evidence
from petzyme.pet_retrieval import load_activity_source, match_pet_assays
from petzyme.privacy import safe_public_text
from petzyme.research_advisor import advise_next_experiment
from petzyme.research_session import new_research_session
from petzyme.ui_adapter import dispatch_research_turn, run_offline
from petzyme.ui_experiments import _measurement_index, combined_experimental_rows

WINDOW = {'temperatures_c': [40.0], 'ph_values': [], 'minimum_ph': None,
          'maximum_ph': 7.0, 'pet_forms': []}


@pytest.fixture(scope='module')
def sources():
    snapshot = load_activity_source()['result']
    observations = parse_nrel(load_nrel()['result'])['result']['observations']
    return snapshot, observations


def test_independent_condition_grid_and_source_counts(sources):
    snapshot, observations = sources
    refs = match_pet_assays(snapshot, {'protein_name': 'ESM050'})
    nrel = [o for o in observations if o['enzyme_identifier'] == 'ESM050']
    keys = {(r['assay']['conditions']['temperature_c'], r['assay']['conditions']['ph'],
             r['assay']['material']['form'], r['assay']['material']['morphology']) for r in refs}
    temps = {x[0] for x in keys}
    phs = {x[1] for x in keys}
    materials = {(x[2], x[3]) for x in keys}
    missing = {(t, p, *m) for t, p, m in product(temps, phs, materials)} - keys
    c = summarize_experimental_coverage('ESM050', nrel, refs)
    assert (len(nrel), len(refs), sum(len(r['assay']['measurements']) for r in refs)) == (10, 16, 64)
    assert c['domain_combinations'] == len(temps) * len(phs) * len(materials) == 16
    assert c['documented_combinations'] == len(keys) == 16
    assert c['missing_in_both_count'] == len(missing) == 0
    assert c['same_publication_assays'] == 16 and c['pazy_publications'] == 1


def test_sparse_grid_and_unknown_metadata_do_not_imply_full_coverage(sources):
    snapshot, observations = sources
    base = observations[0]
    sparse = [{**base, 'temperature_c': t, 'ph': ph, 'pet_form': form} for t, ph, form in [
        (40.0, 5.0, 'AMORPHOUS_FILM'), (60.0, 5.0, 'AMORPHOUS_FILM'),
        (40.0, 6.0, 'CRYSTALLINE_POWDER')]]
    coverage = summarize_experimental_coverage('Synthetic', sparse, [])
    assert (coverage['domain_combinations'], coverage['documented_combinations'], coverage['missing_in_both_count']) == (8, 3, 5)
    assert len(coverage['missing_in_both']) == 5
    reference = deepcopy(match_pet_assays(snapshot, {'protein_name': 'ESM050'})[0])
    reference['assay']['conditions']['ph'] = None
    coverage = summarize_experimental_coverage('Synthetic', [], [reference])
    assert coverage['domain_combinations'] == 0
    assert coverage['records_with_unknown_condition'] == 1
    assert coverage['missing_in_both_count'] == 0  # No complete grid is defined.


def test_source_rows_and_individual_measurements_remain_separate(sources):
    snapshot, observations = sources
    cell = next(o for o in observations if o['enzyme_identifier'] == 'ESM050'
                and o['temperature_c'] == 40 and o['ph'] == 6.5 and o['pet_form'] == 'AMORPHOUS_FILM')
    unit = standardized_nrel_values([cell])[0]
    assert cell['original_text'] == '0.02' and cell['unit_status'] == 'UNKNOWN/UNVERIFIED'
    assert unit['status'] == 'INFERRED_SUPPORTED' and unit['standardized_value'] == 20.0
    refs = match_pet_assays(snapshot, {'protein_name': 'ESM050', 'temperature_c': 40.0, 'ph': 6.5,
                                      'form': 'film', 'morphology': 'amorphous'})
    assert len(refs) == 1 and len(refs[0]['assay']['measurements']) == 4
    source_products = [m['value'] for m in refs[0]['assay']['measurements'] if m['metric_type'] == 'specific_activity']
    result = investigate_pet_evidence({'protein_name': 'ESM050', 'temperature_c': 40.0, 'ph': 6.5,
                                       'form': 'film', 'morphology': 'amorphous'})['result']
    assert result['pazy']['total_matching_measurements'] == 4
    preview = preview_pet_evidence(result, safe_public_text)
    rows = combined_experimental_rows(preview)
    products = [r for r in rows if r.source.startswith('PAZy') and r.definition.startswith('specific_activity')]
    assert len(products) == len(source_products) == 2
    assert [float(r.original_response) for r in products] == source_products
    assert len({_measurement_index(r) for r in products}) == 2
    assert all(r.reported_replicate_n is not None for r in products)
    assert all(r.row_linkage for r in products)
    assert all('SOURCE' not in r.definition for r in products)


@pytest.mark.parametrize('enzyme,goal,expected', [
    ('ESM050', 'VERIFY_ZERO', 'VERIFY_RECORDED_ZERO'),
    ('ESM050', 'VERIFY_POSITIVE', 'REPLICATE_POSITIVE_RESPONSE'),
    ('DP003', 'VERIFY_ZERO', 'VERIFY_RECORDED_ZERO'),
    ('DP043', 'VERIFY_POSITIVE', 'REPLICATE_POSITIVE_RESPONSE'),
])
def test_objective_specific_targets_are_source_bound_and_within_window(sources, enzyme, goal, expected):
    snapshot, _ = sources
    result = advise_next_experiment(enzyme, source_scope='PAZY_ONLY', request_window=WINDOW, planning_goal=goal)
    actions = result['result']['plan_result']['result']['recommendations']
    proposals = [a for a in actions if a['code'] == expected]
    assert proposals and all(a['evidence_category'] == 'HYPOTHETICAL' for a in proposals)
    for action in proposals:
        c = action['target_conditions'][0]
        assert c['temperature_c'] == 40 and c['ph'] <= 7
        form, morphology = ('film', 'amorphous') if c['pet_form'] == 'AMORPHOUS_FILM' else ('powder', 'crystalline')
        refs = match_pet_assays(snapshot, {'protein_name': enzyme, 'temperature_c': c['temperature_c'],
                                         'ph': c['ph'], 'form': form, 'morphology': morphology})
        values = [m['value'] for r in refs for m in r['assay']['measurements']
                  if m['metric_type'] == 'specific_activity' and isinstance(m.get('value'), (int, float))]
        assert any(v == 0 if goal == 'VERIFY_ZERO' else v > 0 for v in values)
        assert action['feasibility']['status'] == 'UNKNOWN'
        assert any(b['category'] == 'MEASURED' and b['sources'] for b in action['evidence_basis'])
    if enzyme == 'ESM050' and goal == 'VERIFY_POSITIVE':
        assert any(a['target_conditions'][0]['ph'] == 6.5 and
                   a['target_conditions'][0]['pet_form'] == 'AMORPHOUS_FILM' for a in proposals)
        assert len({a['action_id'] for a in proposals}) == len(proposals)
        assert proposals[0]['action_id'] == 'coverage-replicate_positive_response'


def test_missing_values_are_never_reported_as_numerical_zero():
    assert reported_response({'value': None}) == 'Not recorded'
    assert reported_response({'value': 0}) == '0'


def test_missing_or_negated_objectives_never_pick_a_zero_by_default():
    assert parse_planning_goal('Do not repeat a zero response') == 'UNSPECIFIED'
    assert parse_planning_goal('Verify both zero and positive responses') == 'UNSPECIFIED'
    assert parse_planning_goal('Verify measurable activity') == 'VERIFY_POSITIVE'
    assert parse_planning_goal('Compare recorded conditions') == 'COMPARE'
    assert parse_planning_goal('Explore untested conditions') == 'EXPLORE'
    result = advise_next_experiment('ESM050', source_scope='PAZY_ONLY', request_window=WINDOW)
    actions = result['result']['plan_result']['result']['recommendations']
    assert all(a['category'] != 'PROPOSED_EXPERIMENT' for a in actions)
    assert {a['action_id'] for a in actions} == {'coverage-verify_identity_and_assay', 'coverage-define_laboratory_window'}


def test_unavailable_positive_response_requests_clarification():
    result = advise_next_experiment('DP003', source_scope='PAZY_ONLY', request_window=WINDOW,
                                    planning_goal='VERIFY_POSITIVE')
    actions = result['result']['plan_result']['result']['recommendations']
    assert any(action['code'] == 'CLARIFY_UNAVAILABLE_TARGET' for action in actions)
    assert not any(action['category'] == 'PROPOSED_EXPERIMENT' for action in actions)


def test_untrusted_model_cannot_override_researcher_goal():
    request = interpret_request('What experiment should I perform next for ESM050?')
    state = AgentState('unused')
    call = {'id': 'x', 'type': 'function', 'function': {'name': 'advise_next_experiment',
            'arguments': json.dumps({'enzyme_identifier': 'ESM050', 'planning_goal': 'VERIFY_POSITIVE'})}}
    event = execute_tool_call(call, state, select_tool_catalog(request))
    assert event['status'] == 'ERROR' and event['error_code'] == 'ARGUMENT_SCHEMA'


def test_multiturn_goals_keep_source_subject_and_constraints():
    session = new_research_session()
    questions = ['Investigate ESM050 using NREL and PAZy.', 'Show PAZy evidence only.',
                 'What should I test next if my laboratory supports only 40C and pH 7 or lower?',
                 'Please verify a recorded zero response.', 'Please replicate a documented positive response.']
    outputs = []
    for question in questions:
        session, result, _ = dispatch_research_turn(session, question, run_offline)
        assert result['result'] is not None
        assert result['result']['synthesis']['status'] == 'VALIDATED_FACT_SELECTION'
        assert session.enzyme_identifiers == ('ESM050',)
        outputs.append(result['result'])
    assert session.sources == ('PAZy',) and session.available_temperatures_c == (40.0,)
    assert session.maximum_ph == 7.0
    for view, code in [(outputs[3], 'VERIFY_RECORDED_ZERO'), (outputs[4], 'REPLICATE_POSITIVE_RESPONSE')]:
        assert any(a['code'] == code for a in view['plan']['recommendations'])
        assert not view['observations']
