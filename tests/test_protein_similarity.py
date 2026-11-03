"""Bounded ranking tests with explicitly synthetic embeddings and CSV fixtures."""
import json
import pytest
from petzyme import protein_similarity as similarity
from petzyme import esm
from petzyme.observations import load_nrel
from petzyme.results import outcome
from test_esm import SyntheticRuntime


@pytest.fixture
def table(tmp_path,monkeypatch):
    path=tmp_path/'synthetic.csv'
    path.write_text(',sequence\nQ,ACDEF\nB,ACDEG\nA,ACDEH\nX,ACDX\nZ,ACDEI\n')
    def embed(prepared):
        return esm.embed_protein(prepared,runtime=SyntheticRuntime())
    monkeypatch.setattr(similarity,'embed_protein',embed)
    return load_nrel(str(path))['result']


def test_ranking_ties_scope_and_self_exclusion(table):
    result=similarity.find_similar_nrel_proteins(table,'Q',top_k=2,max_candidates=3)
    assert result['status']=='SUCCESS' and result['evidence_category']=='COMPUTED'
    value=result['result']; c=value['coverage']
    assert [n['identifier'] for n in value['neighbors']]==['A','B']
    assert all(n['cosine_similarity']==pytest.approx(1) for n in value['neighbors'])
    assert c['total_records']==5 and c['candidate_window_records']==3
    assert c['eligible_candidates']==c['searched_candidates']==2
    assert c['excluded_in_window']==1 and c['outside_window_records']==1 and c['query_excluded_records']==1
    assert c['scope']=='BOUNDED_CANDIDATE_WINDOW'
    assert value['exclusions'][0]['reason']=='AMBIGUOUS_RESIDUES'
    assert value['query_source_row']==2
    assert value['neighbors'][0]['provenance'][0]['name']=='NREL'
    serialized=json.dumps(result,allow_nan=False)
    assert 'ACDEF' not in serialized and 'vector' not in serialized and 'normalized_sequence' not in serialized


@pytest.mark.parametrize('kwargs',[{'top_k':0},{'top_k':True},{'max_candidates':0},{'max_candidates':33},{'top_k':11}])
def test_bounded_arguments(table,kwargs):
    assert similarity.find_similar_nrel_proteins(table,'Q',**kwargs)['status']=='ERROR'


def test_missing_ambiguous_or_invalid_query(table):
    assert similarity.find_similar_nrel_proteins(table,'missing')['status']=='NO_DATA'
    assert similarity.find_similar_nrel_proteins(table,'X')['status']=='ERROR'
    table['rows'].append(table['rows'][0])
    assert similarity.find_similar_nrel_proteins(table,'Q')['status']=='ERROR'


def test_candidate_duplicate_identifiers_and_missing_sequence(table):
    table['rows'][1][0]='A'
    table['rows'][3][1]=''
    result=similarity.find_similar_nrel_proteins(table,'Q',max_candidates=4)['result']
    assert result['coverage']['searched_candidates']==1
    assert [e['reason'] for e in result['exclusions']]==['AMBIGUOUS_OR_MISSING_IDENTIFIER']*2+['MISSING']


def test_query_model_failure_propagates_without_fake_embedding(table,monkeypatch):
    monkeypatch.setattr(similarity,'embed_protein',lambda p: outcome('ERROR',None,p['provenance'],[], 'COMPUTED','model unavailable'))
    result=similarity.find_similar_nrel_proteins(table,'Q')
    assert result['status']=='ERROR' and result['result'] is None


def test_partial_candidate_failure_and_window_bound(table,monkeypatch):
    original=similarity.embed_protein
    seen=[]
    def embed(prepared):
        seen.append(prepared['identifier'])
        if prepared['identifier']=='B': return outcome('ERROR',None,prepared['provenance'],[],'COMPUTED','test failure')
        return original(prepared)
    monkeypatch.setattr(similarity,'embed_protein',embed)
    result=similarity.find_similar_nrel_proteins(table,'Q',max_candidates=2)
    assert result['status']=='PARTIAL'
    assert seen==['Q','B','A']
    assert result['result']['coverage']['eligible_candidates']==2
    assert result['result']['coverage']['searched_candidates']==1
    assert result['result']['exclusions'][0]['reason']=='INFERENCE_FAILED'


def test_no_eligible_candidates_still_reports_coverage(table):
    table['rows']=table['rows'][:1]+[['BAD','AX']]
    result=similarity.find_similar_nrel_proteins(table,'Q')
    assert result['status']=='NO_DATA' and result['result']['neighbors']==[]
    assert result['result']['coverage']['excluded_in_window']==1
