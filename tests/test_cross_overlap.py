import builtins
import io
import json
import logging
from copy import deepcopy

import httpx
import pytest

from test_activity_snapshot import records, load
from zymecycle.cross_overlap import candidate_protein_pairs, identify_cross_source_overlap, verify_snapshot_identities
from zymecycle.cross_comparability import describe_nrel_observation, describe_pazy_assay
from zymecycle.observations import load_nrel, parse_nrel
from zymecycle.pazy import PazyClient


@pytest.fixture
def inputs(tmp_path,records):
    p=tmp_path/'nrel.csv';p.write_text(',sequence,activity_at_6.5_40_aFilm,activity_at_6.5_60_cryPow\nSYNTHETIC,ACDEF,0,1\n')
    records[0]['literature_doi']='10.1021/acscatal.5c03460'
    return load(tmp_path,records)['result'],load_nrel(str(p))['result']


def api(sequence,calls,status=200):
    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(status,json={'id':1,'name':'PRIVATE_API_NAME','amino_acid_sequence':sequence,'literature':[]})
    return PazyClient(transport=httpx.MockTransport(respond))


def test_name_condition_and_publication_overlap_not_verified_identity(inputs):
    snapshot,nrel=inputs
    result=identify_cross_source_overlap(snapshot,nrel)['result']
    assert result['candidate_pair_count']==1 and result['verified_candidate_pairs']==0
    assert result['target_publication_parents']==1 and result['target_publication_assays']==1
    assert result['name_condition_pairs']==1 and result['name_condition_matched_nrel_cells']==1
    assert result['comparison_categories']['INSUFFICIENT_INFORMATION']==1
    assert result['verified_duplicate_cross_source_experiments']==0


@pytest.mark.parametrize('sequence,expected',[('ACDEF','EXACT_SEQUENCE_MATCH'),('ACDEG','DIFFERENT_SEQUENCE'),
                                           (None,'INSUFFICIENT_INFORMATION')])
def test_real_identity_tool_not_name_matching(inputs,sequence,expected):
    snapshot,nrel=inputs;calls=[]
    proof=verify_snapshot_identities(snapshot,nrel,[(1,'SYNTHETIC')],client=api(sequence,calls))
    assert proof['result'][0]['identity']==expected and calls==['/api/proteins/1/']
    text=json.dumps(proof)
    assert 'ACDEF' not in text and 'PRIVATE_API_NAME' not in text and 'sequence_sha256' not in text
    result=identify_cross_source_overlap(snapshot,nrel,proof['result'])['result']
    assert result['verification_outcomes']=={expected:1}
    if expected=='EXACT_SEQUENCE_MATCH':assert result['comparison_categories']['LIMITED_COMPARISON']==1
    elif expected=='DIFFERENT_SEQUENCE':assert result['comparison_categories']['NOT_COMPARABLE']==1


def test_no_network_outside_small_snapshot_candidates(inputs):
    snapshot,nrel=inputs;calls=[];client=api('ACDEF',calls)
    for candidates in ([],[(999,'SYNTHETIC')],[(1,'SYNTHETIC')]*4):
        assert verify_snapshot_identities(snapshot,nrel,candidates,client=client)['status']=='ERROR'
    assert calls==[]


def test_live_failure_no_fallback_or_followup(inputs):
    snapshot,nrel=inputs;calls=[]
    proof=verify_snapshot_identities(snapshot,nrel,[(1,'SYNTHETIC')],client=api(None,calls,503))
    assert proof['status']=='PARTIAL' and proof['result']==[] and len(calls)==1


def test_source_values_and_metadata_are_not_filled(inputs):
    snapshot,nrel=inputs
    from zymecycle.activity_snapshot import iter_assays
    reference=next(iter_assays(snapshot));before=deepcopy(reference)
    descriptor=describe_pazy_assay(reference)
    assert descriptor['fields']['crystallinity_percent'] is None
    assert descriptor['fields']['replicate_n']=='2–3'
    assert reference==before
    observation=parse_nrel(nrel)['result']['observations'][0]
    left=describe_nrel_observation(observation)
    assert 'measurement_unit' not in left['fields'] and left['publication_doi'] is None


def test_ambiguous_alias_candidates_remain_unresolved(inputs):
    snapshot,nrel=inputs
    nrel['rows'].append(['OTHER','ACDEF','1',''])
    snapshot['records'][0]['synonyms']=[{'name':'OTHER'}]
    result=identify_cross_source_overlap(snapshot,nrel)['result']
    assert result['candidate_pair_count']==2 and result['ambiguous_name_candidate_proteins']==1
    assert result['verified_candidate_pairs']==0


def test_unbound_or_duplicate_proof_is_rejected(inputs):
    snapshot,nrel=inputs
    wrong={'pazy_protein_id':99,'nrel_identifier':'SYNTHETIC','identity':'EXACT_SEQUENCE_MATCH','provenance':[]}
    assert identify_cross_source_overlap(snapshot,nrel,[wrong])['status']=='ERROR'


def test_pure_audit_no_writes_logging_or_persistent_live_cache(inputs,monkeypatch,capsys,caplog):
    snapshot,nrel=inputs;calls=[];client=api('ACDEF',calls)
    def forbidden(*a,**kw):pytest.fail('Unexpected filesystem access')
    with monkeypatch.context() as guard:
        guard.setattr(builtins,'open',forbidden);guard.setattr(io,'open',forbidden)
        with caplog.at_level(logging.DEBUG):
            for _ in range(2):
                proof=verify_snapshot_identities(snapshot,nrel,[(1,'SYNTHETIC')],client=client)
                identify_cross_source_overlap(snapshot,nrel,proof['result'])
    assert len(calls)==2 and set(vars(client))=={'_transport','_last_request'}
    output=capsys.readouterr()
    assert 'PRIVATE_API_NAME' not in output.out+output.err+caplog.text
