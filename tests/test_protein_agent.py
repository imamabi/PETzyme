"""Mock Meta decisions and synthetic embeddings; no paid model or live source requests."""
import json
import subprocess
import sys
import pytest
import httpx
from petzyme import esm
from petzyme.agent import _outbound_safe, run_investigation
from petzyme.agent_types import AgentState
from petzyme.agent_request import interpret_request, select_tool_catalog
from petzyme.agent_reporting import format_research_report
from petzyme.meta import MetaModelClient
from petzyme.tools import invoke_tool, describe_tools
from test_esm import SyntheticRuntime

QUESTION = 'Find proteins with ESM-2 representations similar to LCC-ICCG.'


@pytest.fixture
def source(tmp_path,monkeypatch):
    path=tmp_path/'synthetic.csv'
    path.write_text(',sequence\nLCC-ICCG,'+'ACDEF'*20+'\nESM050,'+'ACDEG'*20+'\nESM051,ACDX\n')
    monkeypatch.setattr(esm,'load_esm2_runtime',lambda:SyntheticRuntime())
    return str(path)


def test_only_high_level_representation_tool_exposed():
    catalog=select_tool_catalog(interpret_request(QUESTION))
    assert {t['function']['name'] for t in catalog}=={'load_nrel','find_similar_nrel_proteins'}
    schema=next(t['function']['parameters'] for t in catalog if t['function']['name']=='find_similar_nrel_proteins')
    assert set(schema['properties'])=={'enzyme_identifier','top_k','max_candidates','table_ref'}
    assert 'table_ref' in schema['required']
    assert len(describe_tools()['result'])==19
    for name in ['embed_protein','prepare_protein_sequence','compare_protein_embeddings']:
        assert invoke_tool(name,{})['status']=='ERROR'


@pytest.mark.parametrize('question',[
    'Find proteins similar to LCC-ICCG.', 'Find similar NREL proteins to LCC-ICCG.',
    'Find similar sequences to LCC-ICCG.', 'Find nearest neighbors of LCC-ICCG.',
])
def test_common_sequence_search_phrasings_route_to_representation_tool(question):
    catalog=select_tool_catalog(interpret_request(question))
    assert {t['function']['name'] for t in catalog}=={'load_nrel','find_similar_nrel_proteins'}


def scripted_http(question=QUESTION,query='LCC-ICCG',omit_evidence=False):
    calls=[]
    def handler(request):
        payload=json.loads(request.content)
        calls.append(payload)
        tools=[m for m in payload['messages'] if m['role']=='tool']
        if not tools: name,args,identifier='load_nrel',{},'p1'
        elif len(tools)==1:
            ref=json.loads(tools[0]['content'])['refs']['table']
            name,args,identifier='find_similar_nrel_proteins',{'table_ref':ref,'enzyme_identifier':query,'top_k':2,'max_candidates':2},'p2'
        else:
            evidence=json.loads(tools[-1]['content'])['facts']
            ids=[f['id'] for f in evidence if f['kind']=='EVIDENCE']
            if omit_evidence: ids=ids[:1]
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'fact_ids':ids})}}]})
        message={'role':'assistant','content':None,'tool_calls':[{'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
        return httpx.Response(200,json={'choices':[{'finish_reason':'tool_calls','message':message}]})
    return handler,calls


def test_mock_meta_routing_coverage_categories_and_no_leakage(source):
    handler,calls=scripted_http()
    result=run_investigation(QUESTION,MetaModelClient(transport=httpx.MockTransport(handler)),csv_path=source)
    assert result['status']=='SUCCESS'
    r=result['result']
    assert r['model_requests']==3 and r['tool_attempts']==2
    assert [e['name'] for e in r['trace']]==['load_nrel','find_similar_nrel_proteins']
    transmitted=json.dumps(calls)
    for forbidden in ['ACDEF'*20,'ACDEG'*20,'normalized_sequence','"vector"','"rows"']:
        assert forbidden not in transmitted
    preview=r['trace'][1]['preview']
    assert preview['coverage']['searched_candidates']==1 and preview['coverage']['excluded_in_window']==1
    assert preview['model']['checkpoint']=='synthetic-test-only'
    assert all(f['evidence_category']!='MEASURED' for f in r['facts'])
    rendered=format_research_report(r)
    assert 'UNKNOWN' in rendered and 'bounded' in rendered and 'cosine similarity' in rendered
    assert all(source['name'] in ('NREL','ESM-2') for f in r['facts'] for source in f['provenance'])


@pytest.mark.parametrize('question,query,issue',[(QUESTION,'ESM050','exact identifier'),
    ('Find similar ESM-2 representations to LCC-ICCG over the entire dataset.','LCC-ICCG','whole-dataset')])
def test_wrong_query_or_entire_dataset_claim_is_partial(source,question,query,issue):
    handler,_=scripted_http(question,query)
    result=run_investigation(question,MetaModelClient(transport=httpx.MockTransport(handler)),csv_path=source)
    assert result['status']=='PARTIAL'
    assert any(issue in s for s in result['result']['validation']['issues'])


def test_omitted_similarity_evidence_rejected(source):
    handler,_=scripted_http(omit_evidence=True)
    # Add a second eligible neighbor, so omitting a fact is observable.
    with open(source,'a') as f: f.write('ESM052,ACDEH\n')
    text=open(source).read().replace('ESM051,ACDX','ESM051,ACDEI')
    with open(source,'w') as f: f.write(text)
    result=run_investigation(QUESTION,MetaModelClient(transport=httpx.MockTransport(handler)),csv_path=source)
    assert result['status']=='PARTIAL'
    assert any('omitted required' in s for s in result['result']['validation']['issues'])


@pytest.mark.parametrize('value',[{'vector':[.1]*480},{'embedding':[1.,2.]},[.1]*480,
    json.dumps({'vector':[.1]*480}),{'amino_acid_sequence':'ACDEF'}, {'rows':[['ACDEF']]}])
def test_outbound_guard_rejects_vectors_and_sequences(value):
    assert not _outbound_safe(value,AgentState('unused'))


def test_dispatcher_rejects_model_provided_records_or_cache_paths(source):
    assert invoke_tool('find_similar_nrel_proteins',{'enzyme_identifier':'LCC-ICCG','table_ref':'invented'})['status']=='ERROR'
    for extra in ['sequence','vector','model_path','client','runtime']:
        assert invoke_tool('find_similar_nrel_proteins',{'table':{},'enzyme_identifier':'LCC-ICCG',extra:'x'})['status']=='ERROR'


def test_agent_cannot_hide_partial_inference(source,monkeypatch):
    calls=0
    class FailCandidate(SyntheticRuntime):
        def encode(self,sequence):
            nonlocal calls
            calls+=1
            if calls>1: raise RuntimeError('synthetic failure')
            return super().encode(sequence)
    monkeypatch.setattr(esm,'load_esm2_runtime',lambda:FailCandidate())
    handler,_=scripted_http()
    result=run_investigation(QUESTION,MetaModelClient(transport=httpx.MockTransport(handler)),csv_path=source)
    assert result['status']=='PARTIAL'
    assert any('candidates failed' in i for i in result['result']['validation']['issues'])


def test_similarity_does_not_satisfy_experimental_conditions(source):
    question='Compare LCC-ICCG with similar ESM-2 proteins at 40°C, pH 6.5, using powder.'
    handler,_=scripted_http(question)
    result=run_investigation(question,MetaModelClient(transport=httpx.MockTransport(handler)),csv_path=source)
    assert result['status']=='PARTIAL'
    assert any('Required evidence steps' in i for i in result['result']['validation']['issues'])


def test_core_import_and_nrel_access_without_protein_dependencies():
    code='''
import sys, socket
class BlockProteinImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch','transformers'):
            raise ImportError('Protein dependency intentionally unavailable')
sys.meta_path.insert(0,BlockProteinImports())
def blocked(*args, **kwargs): raise AssertionError('Network access forbidden')
socket.socket.connect=blocked
from petzyme.tools import describe_tools
from petzyme.observations import load_nrel
assert len(describe_tools()['result'])==19
assert load_nrel()['status']=='SUCCESS'
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=False)
    assert result.returncode==0, result.stderr
