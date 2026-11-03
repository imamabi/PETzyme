from copy import deepcopy

import pytest

from petzyme.agent_evidence import safe_text
from petzyme.agent_request import interpret_request
from petzyme.pet_accumulation import accumulate_pet_results, requested_condition_coverage
from petzyme.pet_evidence import investigate_pet_evidence
from petzyme.pet_presenter import preview_pet_evidence

QUESTION = 'Compare multi-study NREL and PAZy evidence for ESM050 at 40°C versus 60°C, pH 6.5, crystalline PET powder.'


@pytest.fixture(scope='module')
def temperature_runs():
    return [{'call_id':f't{i}', 'preview':preview_pet_evidence(investigate_pet_evidence({
        'protein_name':'ESM050','temperature_c':t,'ph':6.5,'form':'powder'})['result'],safe_text)}
        for i,t in enumerate((40,60))]


def test_real_temperature_union_keeps_values_and_source_identity(temperature_runs):
    original=deepcopy(temperature_runs)
    result=accumulate_pet_results(temperature_runs,interpret_request(QUESTION))
    a=result['accumulation']
    assert a['condition_coverage']['complete'] and a['condition_coverage']['covered']==2
    assert all(c['conditions']['covered']==1 for c in a['calls'])
    assert [o['original_text'] for o in result['nrel']['observations']]==['0.08499999999999999','0.13']
    assert a['unique_nrel_retrieved']==a['unique_pazy_assays_retrieved']==2
    assert a['unique_pazy_measurements_retrieved']==6 and result['pazy']['total_matching_measurements'] is None
    assert all(c['pazy']['total_matching_measurements']==4 for c in a['calls'])
    assert all(c['assays'][0]['omitted_measurements']==1 for c in a['calls'])
    assert not result['quantitative_superiority_supported'] and not result['causal_claim_supported']
    assert result['identity']['status']=='PER_CALL_ASSESSMENTS' and not result['identity']['historical_checks_reused']
    assert temperature_runs==original


def test_duplicate_source_evidence_does_not_multiply_counts(temperature_runs):
    runs=[*temperature_runs,{'call_id':'repeated','preview':temperature_runs[0]['preview']}]
    a=accumulate_pet_results(runs,interpret_request(QUESTION))['accumulation']
    assert a['run_count']==3 and a['unique_nrel_retrieved']==2
    assert a['unique_pazy_assays_retrieved']==2 and a['unique_pazy_measurements_retrieved']==6
    assert a['unique_comparisons_retrieved']==6 and a['source_conflicts']==0


def test_condition_grid_requires_actual_tuples_not_marginal_values(temperature_runs):
    observations=deepcopy([r['preview']['nrel']['observations'][0] for r in temperature_runs])
    observations[1]['ph']=8
    request=interpret_request(QUESTION+' Also pH 8.')
    c=requested_condition_coverage(request,observations)
    assert c['requested']==4 and c['covered']==2 and not c['complete']
    assert requested_condition_coverage(request,[])['covered']==0


def test_conflicting_source_cell_is_flagged_not_silently_deduplicated(temperature_runs):
    conflicting=deepcopy(temperature_runs[0])
    conflicting['preview']['nrel']['observations'][0]['original_text']='SYNTHETIC_CONFLICT'
    r=accumulate_pet_results([temperature_runs[0],conflicting],interpret_request(QUESTION))
    assert r['accumulation']['source_conflicts']==1 and len(r['nrel']['observations'])==2
    assert r['accumulation']['unique_nrel_retrieved']==1
    assert not r['selection_complete']


@pytest.mark.parametrize('source',['pazy','nrel'])
def test_different_source_versions_cannot_be_deduplicated(temperature_runs,source):
    runs=deepcopy(temperature_runs)
    if source=='pazy': runs[1]['preview']['pazy']['archive_sha256']='SYNTHETIC_OTHER_VERSION'
    else: runs[1]['preview']['nrel']['observations'][0]['csv_sha256']='SYNTHETIC_OTHER_VERSION'
    with pytest.raises(ValueError): accumulate_pet_results(runs,interpret_request(QUESTION))


def test_collection_and_condition_summary_are_bounded(temperature_runs):
    with pytest.raises(ValueError): accumulate_pet_results(temperature_runs*5,interpret_request(QUESTION))
    request=interpret_request(QUESTION)
    request['temperatures']=list(range(70))
    c=requested_condition_coverage(request,[])
    assert len(c['missing_conditions'])==64 and c['unlisted_conditions']==6 and not c['complete']


def test_synthetic_display_bounds_preserve_omission_counts(temperature_runs):
    # Synthetic source positions exercise cardinality only; these are not scientific findings.
    runs=[]
    for i in range(8):
        run=deepcopy(temperature_runs[0]);run['call_id']=f'synthetic{i}'
        run['preview']['nrel']['observations'][0]['source_row']=1000+i
        card=run['preview']['pazy']['records'][0]
        card['locator']=f'SYNTHETIC_ASSAY_{i}'
        for j,m in enumerate(card['measurements']):m['locator']=f'SYNTHETIC_ASSAY_{i}_MEASUREMENT_{j}'
        run['preview']['comparisons']=[]
        runs.append(run)
    value=accumulate_pet_results(runs,interpret_request(QUESTION));a=value['accumulation']
    assert a['unique_nrel_retrieved']==a['unique_pazy_assays_retrieved']==8
    assert len(value['nrel']['observations'])==len(value['pazy']['records'])==6
    assert a['display_omitted_nrel']==a['display_omitted_assays']==2
    assert a['unique_pazy_measurements_retrieved']==24 and a['display_omitted_measurements']==6
    assert not value['selection_complete']


def test_disjoint_real_measurement_selections_merge_without_inventing_totals(temperature_runs):
    extra=preview_pet_evidence(investigate_pet_evidence({'protein_name':'ESM050','temperature_c':40,
        'ph':6.5,'form':'powder','measurement_index':3})['result'],safe_text)
    value=accumulate_pet_results([temperature_runs[0],{'call_id':'extra','preview':extra}],interpret_request(QUESTION))
    a=value['accumulation']
    assert a['unique_pazy_assays_retrieved']==1 and a['unique_pazy_measurements_retrieved']==4
    assert len(value['pazy']['records'][0]['measurements'])==3 and a['display_omitted_measurements']==1
    assert value['pazy']['records'][0]['omitted_measurements'] is None
    assert a['calls'][0]['assays'][0]['omitted_measurements']==1
    assert a['calls'][1]['assays'][0]['out_of_scope_measurements']==3


def test_synthetic_identity_status_changes_remain_call_bound(temperature_runs):
    second=deepcopy(temperature_runs[0]);second['call_id']='synthetic_proof_scope'
    second['preview']['identity'].update(verified=1,status='EXACT_SEQUENCE_MATCH')
    second['preview']['pazy']['records'][0]['identity_status']='EXACT_SEQUENCE_MATCH'
    value=accumulate_pet_results([temperature_runs[0],second],interpret_request(QUESTION))
    a=value['accumulation']
    assert a['unique_pazy_assays_retrieved']==1 and a['source_conflicts']==0
    assert [c['identity']['verified'] for c in a['calls']]==[0,1]
    assert value['identity']['status']=='PER_CALL_ASSESSMENTS' and value['identity']['verified'] is None
