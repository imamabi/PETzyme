import gzip
import hashlib
import json
from copy import deepcopy

import pytest

from zymecycle import activity_snapshot as module
from zymecycle.activity_snapshot import iter_assays, load_pazy_activity_snapshot


@pytest.fixture
def records():
    return [{'schema_version':'4.2','protein_id':1,'protein_name':'SYNTHETIC', 'literature_doi':'10.1234/synthetic',
             'reference_role':'activity_evidence','synonyms':[], 'supporting_types':[],
             'assays':[{'substrate':'PET','is_representative':True,'representative_group':None,
                        'material':{'form':'film','crystallinity_percent':None,'pretreatment_steps':[]},
                        'conditions':{'temperature_c':40,'ph':6.5,'enzyme_features':[]},
                        'degradation_chemistry':[], 'detection_methods':[],
                        'measurements':[{'value':0,'unit':'synthetic','metric_type':'synthetic',
                                         'analyte':None,'replicate_n':'2–3','value_is_digitized':False}],
                        'evidence':{'source':'synthetic','location':'Table 1'}}]}]


def load(tmp_path,records,**kwargs):
    data = gzip.compress(json.dumps(records).encode())
    path = tmp_path/'synthetic.json.gz'
    path.write_bytes(data)
    return load_pazy_activity_snapshot(str(path),expected_sha256=hashlib.sha256(data).hexdigest(),expected_counts=None,**kwargs)


def test_source_integrity_and_preserved_shapes(tmp_path,records):
    before=deepcopy(records)
    result=load(tmp_path,records)
    assert result['status']=='SUCCESS' and result['result']['records']==before
    assert result['result']['counts']=={'parents':1,'assays':1,'representative_assays':1,'measurements':1}
    assert result['evidence_category'] is None
    item=list(iter_assays(result['result']))[0]
    assert item['assay']['measurements'][0]['value']==0
    assert item['assay']['measurements'][0]['replicate_n']=='2–3'
    assert item['source']['locator'].endswith('#/0/assays/0')
    assert list(iter_assays(result['result'],'PLA'))==[]
    json.dumps(result,allow_nan=False)


def test_checksum_counts_and_corrupt_archive_fail_closed(tmp_path,records):
    path=tmp_path/'bad.gz';path.write_bytes(gzip.compress(json.dumps(records).encode()))
    assert load_pazy_activity_snapshot(str(path))['status']=='ERROR'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_pazy_activity_snapshot(str(path),expected_sha256=digest)['status']=='ERROR'
    path.write_bytes(b'invalid gzip')
    assert load_pazy_activity_snapshot(str(path),expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())['status']=='ERROR'


@pytest.mark.parametrize('change',[{'schema_version':'5'}, {'protein_id':True}, {'amino_acid_sequence':'FORBIDDEN'},
    {'assays':None}, {'literature_doi':'10.0000/example'}])
def test_wrong_schema_and_illustrative_records_rejected(tmp_path,records,change):
    records[0].update(change)
    assert load(tmp_path,records)['status']=='ERROR'


@pytest.mark.parametrize('field,value',[('value',True),('value',float('nan')),('value',float('inf')),
    ('value','1'),('value_is_digitized',1),('unrecognized',None)])
def test_bad_measurements_rejected(tmp_path,records,field,value):
    records[0]['assays'][0]['measurements'][0][field]=value
    assert load(tmp_path,records)['status']=='ERROR'


def test_decompression_and_compressed_bounds(tmp_path,records,monkeypatch):
    monkeypatch.setattr(module,'MAX_JSON',20)
    assert load(tmp_path,records)['status']=='ERROR'
    monkeypatch.setattr(module,'MAX_COMPRESSED',1)
    assert load(tmp_path,records)['status']=='ERROR'


def test_duplicate_json_keys_not_silently_lost(tmp_path):
    raw=b'[{"schema_version":"4.2","schema_version":"4.2"}]'
    data=gzip.compress(raw);p=tmp_path/'duplicate.gz';p.write_bytes(data)
    assert load_pazy_activity_snapshot(str(p),expected_sha256=hashlib.sha256(data).hexdigest(),expected_counts=None)['status']=='ERROR'
