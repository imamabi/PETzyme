from copy import deepcopy
import json

import pytest

from test_activity_snapshot import records, load
from zymecycle.pet_retrieval import load_activity_source, match_pet_assays, retrieve_pet_assays


@pytest.fixture(scope='module')
def real_snapshot():
    result = load_activity_source()
    assert result['status'] == 'SUCCESS', 'Required pinned local scientific source must be available'
    return result['result']


def test_real_exact_identifiers_and_publication(real_snapshot):
    result = retrieve_pet_assays(real_snapshot, {'protein_name':'ESM050'})
    assert result['result']['total_matches'] > 3 and result['status'] == 'PARTIAL'
    card = result['result']['records'][0]
    query = {k:card[k] for k in ('parent_index','assay_index','protein_id','publication_doi')}
    exact = retrieve_pet_assays(real_snapshot, query)['result']
    assert exact['total_matches'] == 1
    assert exact['records'][0] == card
    assert card['source']['locator'].endswith(card['locator'])
    assert card['identity_status'] == 'UNVERIFIED'
    assert retrieve_pet_assays(real_snapshot, {'publication_doi':card['publication_doi'].upper()})['result']['total_matches'] > 0


def test_real_lcc_iccg_exact_name_gap_is_not_an_absence_claim(real_snapshot):
    result=retrieve_pet_assays(real_snapshot,{'protein_name':'LCC-ICCG'})
    assert result['status']=='NO_DATA' and result['result']['total_matches']==0
    assert any('unresolved candidates' in v for v in result['limitations'])


def test_real_all_context_filters_are_exact(real_snapshot):
    ref = match_pet_assays(real_snapshot, {'protein_name':'ESM050','temperature_c':40,'ph':6.5,'form':'powder'})[0]
    a = ref['assay']
    query = {'parent_index':ref['parent_index'],'assay_index':ref['assay_index'], 'measurement_index':0}
    for key in ('duration_value','duration_unit','buffer','reaction_volume_value','reaction_volume_unit'):
        if a['conditions'].get(key) is not None: query[key] = a['conditions'][key]
    result = retrieve_pet_assays(real_snapshot, query)['result']
    m = result['records'][0]['measurements'][0]
    assert m['record'] == a['measurements'][0]
    assert result['records'][0]['context']['conditions'] == a['conditions']
    assert m['locator'].endswith('/measurements/0')
    assert result['total_matches'] == 1
    assert result['total_matching_measurements']==1 and not result['truncated']
    assert result['records'][0]['omitted_measurements']==0
    assert result['records'][0]['out_of_scope_measurements']==len(a['measurements'])-1
    assert retrieve_pet_assays(real_snapshot, {**query,'temperature_c':-999})['status']=='NO_DATA'


def test_real_bounds_parent_assay_measurement_distinction(real_snapshot):
    result = retrieve_pet_assays(real_snapshot)['result']
    assert result['total_matches'] == 5253 and result['total_matching_measurements'] == 8650
    assert result['matching_parent_records'] == 367 and result['matching_publications'] == 91
    assert result['returned'] == 3 and result['truncated']
    assert len({r['publication_doi'] for r in result['records']}) == 3
    assert all(len(r['measurements']) <= 3 for r in result['records'])
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('query,limit', [({'unknown':'x'},3),({'protein_id':True},3),({'assay_index':-1},3),
    ({'temperature_c':float('nan')},3),({'form':''},3),({},0),({},4),({},True)])
def test_invalid_contracts_fail_without_source_access(query,limit):
    assert retrieve_pet_assays({},query,limit)['status']=='ERROR'


def test_synthetic_ambiguity_missingness_units_and_qualifiers(tmp_path, records):
    other=deepcopy(records[0]);other['protein_id']=2;other['synonyms']=[{'name':'ALIAS'}]
    records[0]['synonyms']=[{'name':'ALIAS'}]
    records.append(other)
    records[0]['assays'][0]['measurements'][0].update(unit='original unit',qualifier='<',value=0)
    snapshot=load(tmp_path,records)['result'];before=deepcopy(snapshot)
    result=retrieve_pet_assays(snapshot,{'protein_name':'ALIAS'})['result']
    assert result['name_ambiguity'] and result['matching_protein_ids']==2
    card=result['records'][0]
    assert card['measurements'][0]['record']['unit']=='original unit'
    assert card['measurements'][0]['record']['qualifier']=='<'
    assert 'conditions.duration_value' in card['missing_fields']
    assert 'uncertainty_value' in card['measurements'][0]['missing_fields']
    assert snapshot==before
    assert retrieve_pet_assays(snapshot,{'duration_value':48})['status']=='NO_DATA'


def test_truncation_counts_measurements_even_one_assay(tmp_path,records):
    a=records[0]['assays'][0]
    a['measurements']*=5
    result=retrieve_pet_assays(load(tmp_path,records)['result'])
    assert result['status']=='PARTIAL' and result['result']['records'][0]['omitted_measurements']==2


def test_projection_and_selection_cannot_bypass_bounds(real_snapshot):
    from zymecycle.pet_retrieval import project_pet_assays,diverse_assays
    for limit in (0,4,True):
        assert project_pet_assays(real_snapshot,[],{},limit)['status']=='ERROR'
        with pytest.raises(ValueError):diverse_assays([],limit)


def test_changed_manifest_pin_fails_closed(tmp_path,monkeypatch):
    from zymecycle import pet_retrieval
    (tmp_path/'PAZY_SOURCE_MANIFEST.json').write_text(json.dumps({'filename':'not-authorized.gz'}))
    monkeypatch.setattr(pet_retrieval,'ROOT',tmp_path)
    assert pet_retrieval.load_activity_source()['status']=='ERROR'
