from copy import deepcopy
import pytest

from petzyme.cross_comparability import REQUIRED, assess_cross_source_comparability

SOURCE={'name':'Synthetic documentation','url':'https://example.invalid/','doi':'10.1234/test',
        'locator':'Table 1, row A, replicate 1','attribution':'Synthetic fixture','retrieved_at_utc':None}


def descriptors():
    fields={'substrate':'PET','temperature_c':40,'ph':6.5,'duration_value':48,'duration_unit':'h',
            'form':'film','morphology':'amorphous','crystallinity_percent':4,
            'pretreatment':'Explicitly documented as untreated','enzyme_amount_value':5,'enzyme_amount_unit':'µg',
            'substrate_amount_value':3.5,'substrate_amount_unit':'mg','reaction_volume_value':.5,'reaction_volume_unit':'mL',
            'enzyme_features':['Unmodified tagless protein, documented'], 'buffer':'50 mM phosphate, pH 6.5',
            'detection_methods':[{'method':'UV-Vis 260/280 nm','role':'quantitative'}],
            'measurement_definition':'aromatic product amount per enzyme mass after 48 h',
            'analyte':'TPA + MHET + BHET','measurement_unit':'µmol/mg','replicate_n':2,
            'uncertainty_value':0,'uncertainty_type':'documented range'}
    assert set(fields)==set(REQUIRED)
    left={'protein_id':'N1','publication_doi':'10.1234/test','experiment_locator':'Table 1, row A, replicate 1',
          'fields':fields,'source':SOURCE}
    right=deepcopy(left);right['protein_id']='7'
    verification={'nrel_identifier':'N1','pazy_protein_id':7,'identity':'EXACT_SEQUENCE_MATCH','provenance':[SOURCE]}
    return left,right,verification


def test_fully_documented_equivalence_is_not_causality():
    left,right,proof=descriptors()
    result=assess_cross_source_comparability(left,right,proof)
    assert result['result']['category']=='VERIFIED_COMPARABLE'
    assert result['result']['duplicate_status']=='SAME_DOCUMENTED_EXPERIMENT'
    assert not result['result']['causal_claim_supported']
    assert result['evidence_category']=='COMPUTED'


@pytest.mark.parametrize('field,value',[('substrate','PLA'),('form','powder'),('temperature_c',60),
    ('ph',7.5),('duration_value',2),('measurement_definition','mass_loss'),('measurement_unit','%'),
    ('detection_methods','HPLC'),('enzyme_amount_value',99)])
def test_documented_incompatibility_is_not_pooled(field,value):
    left,right,proof=descriptors();right['fields'][field]=value
    result=assess_cross_source_comparability(left,right,proof)['result']
    assert result['category']=='NOT_COMPARABLE' and field in result['incompatible_fields']


def test_missing_protocol_is_limited_and_shared_doi_not_duplicate():
    left,right,proof=descriptors();left['fields']['duration_value']=None
    result=assess_cross_source_comparability(left,right,proof)['result']
    assert result['category']=='LIMITED_COMPARISON' and result['duplicate_status']=='SHARED_PUBLICATION_ONLY'


@pytest.mark.parametrize('identity',['DIFFERENT_SEQUENCE','INSUFFICIENT_INFORMATION','EXACT_SEQUENCE_MATCH'])
def test_identity_proof_must_bind_to_both_records(identity):
    left,right,proof=descriptors();proof['identity']=identity;proof['pazy_protein_id']=999
    assert assess_cross_source_comparability(left,right,proof)['result']['category']=='INSUFFICIENT_INFORMATION'


def test_unknown_identity_and_no_measurement_remain_insufficient():
    left,right,_=descriptors();right['fields']['numeric_response_available']=False
    assert assess_cross_source_comparability(left,right)['result']['category']=='INSUFFICIENT_INFORMATION'


def test_censored_measurement_does_not_establish_direct_equivalence():
    left,right,proof=descriptors();right['fields']['qualifier']='<'
    assert assess_cross_source_comparability(left,right,proof)['result']['category']=='LIMITED_COMPARISON'
