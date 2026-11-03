from copy import deepcopy

from scripts.audit_nrel_units import reconcile_nrel


def row(identifier, value, number=2, temperature=40):
    return {'row':number,'sample_id':identifier,'pH':6.5,'temperature_c':temperature,'substrate':'cryPow',
            'reaction_vol_mL':0.5,'mM_product_per_mg_enzyme':value*2000,
            'mmol_product_per_mg_enzyme':value, 'umol_product_per_mg_enzyme':value*1000}


def test_audit_distinguishes_exact_rounding_consistency_and_exception():
    csv = ',activity_at_6.5_40_cryPow,activity_at_6.5_60_cryPow\nE1,0.085,\nE2,0.01,\nE3,0.01,\nE4,0,\n'
    activity = [row('E1',.08),row('E1',.09,3),row('E2',.006,4),row('E3',.1,5),row('E4',0,6)]
    original = deepcopy(activity)
    result = reconcile_nrel(csv,activity)
    assert result['counts'] == {'observations':4,'matched':4,'unmatched':0,'exact_mean_matches':2,
                               'within_rounding_interval':3,'outside_rounding_interval':1,'workbook_scale_consistent_rows':5,
                               'reconstructed_matches':3,'reconstructed_mismatches':1,'rounded_V_mean_matches':3}
    assert result['exceptions'][0]['identifier']=='E3'
    assert result['unit_status']=='UNKNOWN/UNVERIFIED' and result['evidence_category']=='COMPUTED'
    assert activity==original


def test_no_matching_condition_is_not_zero_and_form_is_exact():
    result = reconcile_nrel(',activity_at_6.5_40_aFilm\nE,0\n', [row('E',0)])
    assert result['counts']['unmatched']==1 and result['counts']['within_rounding_interval']==0


def test_exception_output_is_bounded():
    csv = ',activity_at_6.5_40_cryPow\n' + ''.join(f'E{i},9\n' for i in range(12))
    result = reconcile_nrel(csv,[row(f'E{i}',1,i+2) for i in range(12)])
    assert len(result['exceptions'])==8 and result['exceptions_omitted']==4
