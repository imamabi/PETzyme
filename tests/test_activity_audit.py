from copy import deepcopy

from test_activity_snapshot import records, load
from petzyme.activity_audit import audit_pazy_pet_assays, audit_publication_workbook_links, field_coverage


def test_nested_counts_pet_filter_and_missing_measurements(tmp_path,records):
    extra=deepcopy(records[0]);extra['protein_id']=2;extra['assays'][0]['substrate']='PLA'
    assays=records[0]['assays']
    assays[0]['measurements'] += [{'value':None,'categorical_result':'positive'},
                                {'reported_result':'Qualitative narrative'}, {},
                                {'value':1.5,'qualifier':'<','value_is_digitized':True}]
    records.append(extra)
    before=deepcopy(records)
    snapshot=load(tmp_path,records)['result']
    result=audit_pazy_pet_assays(snapshot)['result']
    assert result['snapshot_counts']['parents']==2 and result['snapshot_counts']['assays']==2
    assert result['pet_assays']==1 and result['pet_measurements']==5
    assert result['pet_proteins']==1 and result['pet_publications']==1
    assert result['measurement_kinds']=={'numeric':2,'categorical_without_numeric':1,'text_only':1,'missing_result':1}
    assert result['numeric_zero_measurements']==1 and result['numeric_with_qualifier']==1
    assert result['textual_replicate_counts']==1
    assert result['digitization']=={'true':1,'false':1,'unavailable':3}
    assert snapshot['records']==before


def test_null_missing_zero_and_empty_are_distinct():
    assert field_coverage([{}, {'x':None},{'x':0},{'x':[]},{'x':False}], 'x') == {
        'absent':1,'null':1,'empty':1,'documented':2}


def test_duplicate_publication_not_duplicate_experiment(tmp_path,records):
    other=deepcopy(records[0]);other['protein_id']=2;records.append(other)
    audit=audit_pazy_pet_assays(load(tmp_path,records)['result'])['result']
    assert audit['pet_publications']==1 and audit['duplicate_parent_key_groups']==0
    assert audit['exact_curation_duplicate_candidate_groups']==0
    records.append(deepcopy(records[0]))
    audit=audit_pazy_pet_assays(load(tmp_path,records)['result'])['result']
    assert audit['duplicate_parent_key_groups']==1 and audit['exact_curation_duplicate_candidate_excess_assays']==1


def test_no_pet_is_explicit_no_data(tmp_path,records):
    records[0]['assays'][0]['substrate']='PLA'
    result=audit_pazy_pet_assays(load(tmp_path,records)['result'])
    assert result['status']=='NO_DATA' and result['result']['pet_assays']==0


def test_workbook_links_validate_conditions_values_and_multiplicity(tmp_path,records):
    records[0]['literature_doi']='10.1021/acscatal.5c03460'
    a=records[0]['assays'][0]
    a['evidence']['location']='cs5c03460_si_002.xlsx, Activity sheet, rows 2, 3; main article'
    a['measurements']=[{'metric_type':'specific_activity','value':1}, {'metric_type':'specific_activity','value':2},
                       {'metric_type':'conversion_percent','value':.1},{'metric_type':'conversion_percent','value':.2}]
    snapshot=load(tmp_path,records)['result']
    rows={i:{'sample_id':'SYNTHETIC','pH':6.5,'temperature_c':40,'substrate':'aFilm',
             'umol_product_per_mg_enzyme':v,'percent_depolymerization':v/10} for i,v in [(2,1),(3,2)]}
    result=audit_publication_workbook_links(snapshot,rows,snapshot['source'])['result']
    assert result['matching_protein_condition_form_assays']==1
    assert result['matching_activity_value_groups']==1 and result['matching_conversion_value_groups']==1
    assert result['unique_referenced_workbook_rows']==2
    rows[3]['umol_product_per_mg_enzyme']=1
    assert audit_publication_workbook_links(snapshot,rows,snapshot['source'])['result']['activity_value_group_mismatches']==1
