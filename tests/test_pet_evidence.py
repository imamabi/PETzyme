import json
import pytest

from zymecycle.pet_evidence import investigate_pet_evidence


def test_real_lcc_question_keeps_exact_gap_and_original_values():
    result=investigate_pet_evidence({'protein_name':'LCC-ICCG'})
    assert result['status']=='PARTIAL'
    v=result['result']
    assert v['nrel']['total_matches']>3 and v['pazy']['total_matches']==0
    assert all(o['enzyme_identifier']=='LCC-ICCG' and o['source_row']==41 for o in v['nrel']['observations'])
    assert v['identity']['verified']==0 and not v['identity']['historical_checks_reused']
    assert 'does not establish absence' in v['answer']
    assert v['units']['csv_units'].startswith('UNRESOLVED original declaration (UNKNOWN/UNVERIFIED)')
    assert 'INFERRED' in v['units']['csv_units'] and 'COMPUTED' in v['units']['csv_units']
    assert all(o['unit_status'] == 'UNKNOWN/UNVERIFIED' for o in v['nrel']['observations'])
    assert 'INFERRED' in v['units']['csv_generation']
    assert result['evidence_category']=='COMPUTED'


def test_real_external_publications_and_conditions_computed_from_sources():
    result=investigate_pet_evidence(scope='OUTSIDE_NREL_PUBLICATION')
    v=result['result']
    assert v['coverage']['pet_assays']==5253 and v['coverage']['pet_publications']==91
    assert v['coverage']['outside_publication_assays']==4169 and v['coverage']['outside_publications']==90
    assert v['coverage']['same_study_assays_without_condensed_cell']==313
    assert v['pazy']['total_matches']==4169 and v['pazy']['returned']==3 and v['pazy']['truncated']
    assert all(c['publication_doi']!='10.1021/acscatal.5c03460' for c in v['pazy']['records'])
    assert len({c['publication_doi'] for c in v['pazy']['records']})==3
    assert v['coverage']['comparison_categories']['VERIFIED_COMPARABLE']==0
    assert v['coverage']['verified_candidate_pairs']==0


def test_real_same_study_not_independent_and_no_scaling():
    v=investigate_pet_evidence({'protein_name':'ESM050','ph':6.5,'form':'powder'})['result']
    assert v['pazy']['total_matches']==2
    assert {o['original_text'] for o in v['nrel']['observations']}=={'0.08499999999999999','0.13'}
    assert v['comparisons'] and all(not c['quantitative_comparison_supported'] for c in v['comparisons'])
    assert any('temperature_c' in c['incompatible_fields'] for c in v['comparisons'])
    assert 'never independent' in v['overlap_interpretation']
    assert not v['quantitative_superiority_supported'] and not v['causal_claim_supported']
    assert all(o['unit_status']=='UNKNOWN/UNVERIFIED' for o in v['nrel']['observations'])
    assert v['gaps'][0]['code']=='VERIFY_PROTEIN_AND_CONSTRUCT'
    assert any(g['code']=='AUTHENTICATE_CSV_UNITS' for g in v['gaps'])
    json.dumps(v,allow_nan=False)


@pytest.mark.parametrize('query,scope',[({'temperature_c':-999},'ALL'),({'form':'nonexistent form'},'ALL')])
def test_real_unmeasured_conditions_not_predicted(query,scope):
    v=investigate_pet_evidence(query,scope=scope)['result']
    assert not v['nrel']['observations'] and not v['pazy']['records']
    assert 'No PAZy assay matched' in v['answer'] and not v['quantitative_superiority_supported']


def test_missing_source_no_network_fallback(monkeypatch):
    from zymecycle.results import outcome
    monkeypatch.setattr('zymecycle.pet_evidence.load_activity_source',lambda:outcome('ERROR',None,[],[],None))
    assert investigate_pet_evidence()['status']=='ERROR'


@pytest.mark.parametrize('args',[{'limit':4},{'limit':True},{'query':{'metadata':'invented'}},{'scope':'INVENTED'}])
def test_invalid_composition_contracts(args):
    assert investigate_pet_evidence(**args)['status']=='ERROR'


@pytest.mark.parametrize('proof',[{}, {'pazy_protein_id':1,'nrel_identifier':'ESM050','identity':'EXACT_SEQUENCE_MATCH','provenance':[]}])
def test_unbound_proof_is_not_a_source_verification(proof):
    assert investigate_pet_evidence({'protein_name':'ESM050'},verification=proof)['status']=='ERROR'


def test_unit_audit_must_match_the_actual_loaded_csv(monkeypatch):
    from zymecycle import pet_evidence
    original=pet_evidence.load_nrel
    def changed(path):
        loaded=original(path)
        loaded['result']['csv_sha256']='synthetic-different-source'
        return loaded
    monkeypatch.setattr(pet_evidence,'load_nrel',changed)
    result=pet_evidence.investigate_pet_evidence({'protein_name':'ESM050'})
    assert result['status']=='ERROR' and result['result'] is None


def test_historical_candidate_subset_is_insufficient_without_record_bound_proofs():
    from collections import Counter
    from zymecycle.pet_retrieval import load_activity_source
    from zymecycle.activity_snapshot import iter_assays
    from zymecycle.observations import load_nrel,parse_nrel
    from zymecycle.cross_overlap import candidate_protein_pairs
    from zymecycle.cross_comparability import assess_cross_source_comparability,describe_nrel_observation,describe_pazy_assay
    s=load_activity_source()['result'];t=load_nrel()['result'];obs=parse_nrel(t)['result']['observations']
    names={'ESM050','DP043','DP003'}
    candidates={pid:name for pid,name in candidate_protein_pairs(s,t) if name in names}
    categories=Counter();otherwise_limited=Counter()
    for ref in iter_assays(s):
        if ref['protein_id'] not in candidates:continue
        name=candidates[ref['protein_id']];a=ref['assay']
        mi=next((i for i,m in enumerate(a['measurements']) if m.get('metric_type')=='specific_activity'),0)
        for o in obs:
            if o['enzyme_identifier']!=name or (o['temperature_c'],o['ph'])!=(a['conditions'].get('temperature_c'),a['conditions'].get('ph')):continue
            result=assess_cross_source_comparability(describe_nrel_observation(o),describe_pazy_assay(ref,mi))['result']
            categories[result['category']]+=1
            assert result['identity_status']=='UNVERIFIED'
            if not result['incompatible_fields']:
                assert result['category']=='INSUFFICIENT_INFORMATION'
                assert 'verified_protein_identity' in result['missing_fields']
                assert not {'substrate','temperature_c','ph','numeric_pazy_response'} & set(result['missing_fields'])
                assert 'measurement_unit' in result['missing_fields']
                otherwise_limited[name]+=1
    # No identity proof is reconstructed or supplied: this verifies the decision branch only.
    assert categories=={'INSUFFICIENT_INFORMATION':27,'NOT_COMPARABLE':20}
    assert otherwise_limited=={'ESM050':10,'DP043':7,'DP003':10}
