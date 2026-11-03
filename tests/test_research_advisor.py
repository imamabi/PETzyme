"""Real local evidence validation; no network, provider or protein inference."""
import json
import pytest
from petzyme import research_advisor as advisor
from petzyme.results import outcome

A = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
B = {**A, 'temperature_c': 60}


@pytest.fixture(scope='module')
def real_advice():
    return advisor.advise_next_experiment('ESM050', A, B)


def test_real_sources_exact_values_and_scientific_categories(real_advice):
    assert real_advice['status'] == 'PARTIAL'
    assert real_advice['evidence_category'] == 'HYPOTHETICAL'
    value = real_advice['result']
    investigation = value['condition_result']['result']
    assert [c['observations'][0]['original_text'] for c in investigation['conditions']] == ['0.08499999999999999', '0.13']
    assert all(c['observations'][0]['unit_status'] == 'UNKNOWN/UNVERIFIED' for c in investigation['conditions'])
    assert not investigation['direct_comparison_supported'] and not investigation['causal_attribution_supported']
    assert {s['name'] for s in real_advice['provenance']} == {'NREL', 'PAZy activity snapshot'}
    assert value['persistence'] == 'TRANSIENT_PAZY_SELECTION'
    json.dumps(real_advice, allow_nan=False)  # In memory only.


def test_bounded_pazy_context_uses_original_publication_not_replication(real_advice):
    evidence = real_advice['result']['evidence_results']
    assert len(evidence) == 2
    for item in evidence:
        value = item['result']
        assert value['pazy']['returned'] == 1
        assert len(value['pazy']['records'][0]['measurements']) <= 3
        assert value['pazy']['records'][0]['publication_doi'].casefold() == '10.1021/acscatal.5c03460'
        assert not value['identity']['historical_checks_reused']
        assert value['identity']['verified'] == 0
        assert not value['quantitative_superiority_supported'] and not value['causal_claim_supported']
    assert 'independent replication' in real_advice['result']['answer']


def test_existing_priority_controls_gaps_and_feasibility_preserved(real_advice):
    value = real_advice['result']
    actions = value['plan_result']['result']['recommendations']
    assert len(actions) == 7
    assert all(c['evidence_category'] == 'HYPOTHETICAL' for c in actions)
    assert actions[0]['category'] == 'EVIDENCE_VERIFICATION'
    experiment = next(c for c in actions if c['code'] == 'CONTROLLED_COMPARISON')
    assert experiment['varied_variables'] == ['temperature']
    assert experiment['prerequisites']
    assert any('blanks' in m for m in experiment['required_measurements'])
    assert any('replication' in m for m in experiment['required_measurements'])
    assert all(c['feasibility']['status'] == 'UNKNOWN' for c in actions)
    assert {'VERIFY_PROTOCOL_AND_GROUP', 'VERIFY_DEFINITION_AND_UNITS', 'VERIFY_REPLICATION_AND_UNCERTAINTY',
            'VERIFY_PUBLICATION_MAPPING'} <= {g['code'] for g in value['gaps']}
    assert {'NREL_COMPARABILITY_UNRESOLVED', 'CROSS_SOURCE_IDENTITY_UNVERIFIED', 'PAZY_PREVIEW_TRUNCATED'} <= set(value['partial_reasons'])


def test_constraints_are_not_invented():
    value = advisor.advise_next_experiment('ESM050', A, B, optional_constraints={'allow_new_measurements': False})['result']
    experiment = value['plan_result']['result']['recommendations'][-1]
    assert experiment['feasibility']['status'] == 'BLOCKED'


def test_missing_condition_not_predicted():
    value = advisor.advise_next_experiment('ESM050', A, {**B, 'temperature_c': 50})['result']
    assert value['condition_result']['result']['conditions'][1]['observations'] == []
    assert 'NREL_REQUESTED_MEASUREMENTS_MISSING' in value['partial_reasons']
    assert any(c['code'] == 'MEASURE_EXACT_CONDITION' for c in value['plan_result']['result']['recommendations'])


def test_pet_forms_remain_separate():
    value = advisor.advise_next_experiment('ESM050', A, {**A, 'pet_form': 'AMORPHOUS_FILM'})['result']
    assert any(g['code'] == 'SEPARATE_PET_FORMS' for g in value['gaps'])
    assert not any(c['code'] == 'CONTROLLED_COMPARISON' for c in value['plan_result']['result']['recommendations'])


def test_missing_pazy_does_not_fabricate_context(monkeypatch):
    monkeypatch.setattr(advisor, 'investigate_pet_evidence', lambda *a, **k: outcome('ERROR', None, [], [], 'COMPUTED'))
    result = advisor.advise_next_experiment('ESM050', A, B)
    assert result['status'] == 'PARTIAL'
    assert 'PAZY_CONTEXT_UNAVAILABLE' in result['result']['partial_reasons']
    assert result['result']['plan_result']['result']['recommendations']


def test_missing_nrel_fails_closed(monkeypatch):
    monkeypatch.setattr(advisor, 'load_nrel', lambda *a: outcome('ERROR', None, [], [], None))
    assert advisor.advise_next_experiment('ESM050', A, B)['status'] == 'ERROR'


def test_repeated_condition_is_not_independent_evidence():
    result = advisor.advise_next_experiment('ESM050', A, A)['result']
    assert len(result['evidence_results']) == 1
    assert any(g['code'] == 'CLARIFY_COMPARISON_TARGET' for g in result['gaps'])


@pytest.mark.parametrize('kwargs', [{'enzyme_identifier': ''}, {'max_recommendations': True},
    {'max_recommendations': 11}, {'optional_constraints': {'instructions': 'invent resources'}},
    {'condition_a': {}}, {'condition_b': {**B, 'ph': float('nan')}}])
def test_invalid_inputs(kwargs):
    assert advisor.advise_next_experiment(**{'enzyme_identifier': 'ESM050', 'condition_a': A, 'condition_b': B, **kwargs})['status'] == 'ERROR'


def test_no_writes_inference_or_api(monkeypatch, capsys, caplog):
    import builtins
    import io
    import os
    from petzyme import esm, pazy, meta
    def blocked(*args, **kwargs): raise AssertionError('Forbidden inference or persistence')
    original = builtins.open
    original_io = io.open
    def read_only(opener):
        def checked(file, mode='r', *args, **kwargs):
            assert not set(mode) & set('wax+'), 'Scientific persistence forbidden'
            return opener(file, mode, *args, **kwargs)
        return checked
    monkeypatch.setattr(builtins, 'open', read_only(original))
    monkeypatch.setattr(io, 'open', read_only(original_io))
    monkeypatch.setattr(os, 'open', blocked)
    monkeypatch.setattr(esm, 'load_esm2_runtime', blocked)
    monkeypatch.setattr(pazy.PazyClient, '_get', blocked, raising=False)
    monkeypatch.setattr(meta.MetaModelClient, 'complete', blocked)
    assert advisor.advise_next_experiment('ESM050', A, B)['result'] is not None
    captured = capsys.readouterr()
    assert not captured.out and not captured.err and not caplog.text
