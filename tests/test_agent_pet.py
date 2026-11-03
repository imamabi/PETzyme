import json

import httpx
import pytest

from petzyme.agent import run_investigation
from petzyme.agent_reporting import format_research_report
from petzyme.agent_request import interpret_request, select_tool_catalog
from petzyme.agent_execution import execute_tool_call
from petzyme.agent_types import AgentState
from petzyme.meta import MetaModelClient
from petzyme.pet_evidence import investigate_pet_evidence
from petzyme.pet_presenter import preview_pet_evidence
from petzyme.agent_evidence import safe_text
from petzyme.ui_adapter import EXAMPLES, run_offline


class PetScenario:
    def __init__(self, arguments=None, final=None):
        self.arguments = arguments if arguments is not None else {'query':{'protein_name':'ESM050','ph':6.5,'form':'powder'}}
        self.final = final
        self.requests = []

    def respond(self,request):
        payload=json.loads(request.content);self.requests.append(payload)
        replies=[json.loads(m['content']) for m in payload['messages'] if m['role']=='tool']
        if not replies:
            message={'role':'assistant','content':None,'tool_calls':[{'id':'pet1','type':'function',
                'function':{'name':'investigate_pet_evidence','arguments':json.dumps(self.arguments)}}]}
            reason='tool_calls'
        else:
            message={'role':'assistant','content':self.final or json.dumps({'fact_ids':[f['id'] for r in replies for f in r['facts']]})}
            reason='stop'
        return httpx.Response(200,json={'choices':[{'finish_reason':reason,'message':message}]})


class LocalPetScenario:
    """In-process acceptance harness; never constructs a provider client or HTTP request."""
    mode='LOCAL_ACCEPTANCE_NO_API'

    def __init__(self,arguments):
        self.arguments=arguments
        self.inputs=[]

    def complete(self,messages,tools):
        self.inputs.append(json.loads(json.dumps(messages)))
        replies=[json.loads(m['content']) for m in messages if m['role']=='tool']
        if not replies:
            return {'role':'assistant','content':None,'tool_calls':[{'id':'local1','type':'function',
                'function':{'name':'investigate_pet_evidence','arguments':json.dumps(self.arguments)}}]}
        return {'role':'assistant','content':json.dumps({'fact_ids':[f['id'] for r in replies for f in r['facts']]})}


def test_real_science_mock_meta_http_canonical_grounding_and_privacy():
    scenario=PetScenario()
    result=run_investigation(EXAMPLES['Multi-study comparison'],MetaModelClient(transport=httpx.MockTransport(scenario.respond)))
    report=result['result']
    assert result['status']=='PARTIAL' and report['validation']['grounded']
    assert report['model_requests']==2 and report['tool_attempts']==1 and len(scenario.requests)==2
    assert report['trace'][0]['name']=='investigate_pet_evidence'
    assert report['persistence']=='FORBIDDEN_PAZY_DERIVED'
    outbound=json.dumps(scenario.requests,allow_nan=False)
    assert all(s not in outbound for s in ('amino_acid_sequence','normalized_sequence','sequence_sha256','"rows"','"vector"','/Users/'))
    assert all(len(json.dumps(r).encode())<262144 for r in scenario.requests)
    preview=report['trace'][0]['preview']
    assert len(preview['pazy']['records'])<=3 and all(len(c['measurements'])<=3 for c in preview['pazy']['records'])
    assert {'0.08499999999999999','0.13'}=={o['original_text'] for o in preview['nrel']['observations']}
    assert all(not c['quantitative_comparison_supported'] for c in preview['comparisons'])
    assert 'PAZy publication-derived assay:' not in format_research_report(report)
    assert 'PAZy publication-derived assay:' in format_research_report(report,public=False)
    assert all(f['ephemeral'] for f in report['facts'])
    assert all(f['provenance'] for f in report['facts'] if f['kind']=='EVIDENCE')


@pytest.mark.parametrize('name',['Multi-study evidence','Outside NREL evidence','Multi-study comparison'])
def test_all_real_offline_demonstrations_are_explicitly_partial(name):
    view=run_offline(EXAMPLES[name])['result']
    assert view['workflow_status']=='PARTIAL' and view['mode']=='DETERMINISTIC_OFFLINE'
    assert view['tool_attempts']==1 and view['model_requests']==2
    assert view['multi_study']['identity']['historical_checks_reused'] is False


def test_least_privilege_catalog_and_bound_proof_only():
    request=interpret_request(EXAMPLES['Multi-study evidence'])
    tools=select_tool_catalog(request)
    assert {t['function']['name'] for t in tools}=={'investigate_pet_evidence', 'advise_next_experiment'}
    assert all(not {'verification', 'metadata', 'csv_path', 'request_window'} & t['function']['parameters']['properties'].keys() for t in tools)
    schema=tools[0]['function']['parameters']
    assert 'verification' not in schema['properties'] and 'verification_ref' in schema['properties']
    state=AgentState('data/nrel/label_data.csv')
    call={'id':'x','function':{'name':'investigate_pet_evidence','arguments':json.dumps({'verification':{'identity':'EXACT_SEQUENCE_MATCH'}})}}
    assert execute_tool_call(call,state,tools)['status']=='ERROR'


def test_nonexistent_proof_handle_is_not_authority():
    scenario=PetScenario({'verification_ref':'invented:proof'})
    report=run_investigation(EXAMPLES['Multi-study evidence'],MetaModelClient(transport=httpx.MockTransport(scenario.respond)))['result']
    assert report['trace'][0]['status']=='ERROR' and report['status']=='PARTIAL'


def test_model_budget_stops_without_retry():
    scenario=PetScenario()
    report=run_investigation(EXAMPLES['Multi-study comparison'],MetaModelClient(transport=httpx.MockTransport(scenario.respond)),max_model_requests=1)['result']
    assert len(scenario.requests)==1 and report['model_requests']==1 and report['tool_attempts']==1
    assert report['status']=='PARTIAL' and any('budget exhausted' in i for i in report['validation']['issues'])


def test_wrong_scope_and_fabricated_superiority_are_rejected():
    scenario=PetScenario({'query':{'protein_name':'ESM050'}},final='ESM050 is better and temperature causes greater activity')
    report=run_investigation(EXAMPLES['Multi-study evidence'],MetaModelClient(transport=httpx.MockTransport(scenario.respond)))['result']
    assert not report['validation']['grounded']
    assert any('requested exact protein' in i for i in report['validation']['issues'])
    assert 'temperature causes greater' not in format_research_report(report,public=False)


def test_projection_filters_source_text_sequences_and_extra_context():
    native=investigate_pet_evidence({'protein_name':'ESM050'})['result']
    card=native['pazy']['records'][0]
    card['context']['conditions']['buffer']='ACDEFGHIKLMNPQRSTVWY'*20
    card['context']['conditions']['provider_response']='PRIVATE_BODY'
    card['context']['material']['unknown_internal']='PRIVATE_BODY'
    preview=preview_pet_evidence(native,safe_text)
    encoded=json.dumps(preview)
    assert 'ACDEFGHIKLMNPQRSTVWY'*20 not in encoded and 'PRIVATE_BODY' not in encoded
    assert preview['source_preview_omissions']
    assert not preview['quantitative_superiority_supported']


@pytest.mark.parametrize('budget',[2,8])
def test_new_tool_obeys_shared_scientific_attempt_budget(monkeypatch,budget):
    from petzyme.results import outcome
    attempts=[]
    def invoke(name,args):
        attempts.append(1)
        return outcome('ERROR',None,[],[],None,'Synthetic failure for budget edge only')
    monkeypatch.setattr('petzyme.agent_execution.invoke_tool',invoke)
    class Batches:
        mode='SYNTHETIC_BUDGET_TEST'
        def complete(self,messages,tools):
            n=len([m for m in messages if m['role']=='assistant'])
            return {'role':'assistant','content':None,'tool_calls':[{'id':f'b{n}_{i}','type':'function',
                'function':{'name':'investigate_pet_evidence','arguments':json.dumps({'query':{'temperature_c':n*10+i}})}} for i in range(8)]}
    report=run_investigation(EXAMPLES['Multi-study evidence'],Batches(),max_tool_calls=budget)['result']
    assert len(attempts)==budget and report['tool_attempts']==budget and report['model_requests']<=6
    assert report['status']=='PARTIAL' and any('budget exhausted' in s for s in report['validation']['issues'])


def test_executed_identity_tool_produces_bound_reusable_transient_handle(monkeypatch):
    # Identity routing edge: the API response is synthetic; no new real PAZy identity is claimed.
    from petzyme.pet_retrieval import ROOT,load_activity_source,match_pet_assays
    from petzyme.observations import load_nrel,select_nrel_protein
    from petzyme.metadata import get_pazy_metadata
    from petzyme.pazy import PazyClient
    table=load_nrel(str(ROOT/'data/nrel/label_data.csv'))['result']
    local=select_nrel_protein(table,'ESM050')['result']
    pid=match_pet_assays(load_activity_source()['result'],{'protein_name':'ESM050'})[0]['protein_id']
    client=PazyClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={
        'id':pid,'name':'SYNTHETIC_IDENTITY_EDGE','amino_acid_sequence':local['amino_acid_sequence']})))
    protein=get_pazy_metadata(pid,client=client)['result']
    state=AgentState(str(ROOT/'data/nrel/label_data.csv'))
    state.artifacts={'n':('nrel_protein',local),'p':('pazy_protein',protein)}
    request=interpret_request('Check exact sequence identity and multi-study evidence for ESM050 across NREL and PAZy.')
    catalog=select_tool_catalog(request)
    event=execute_tool_call({'id':'identity1','function':{'name':'compare_identity','arguments':json.dumps({
        'nrel_protein_ref':'n','pazy_protein_ref':'p'})}},state,catalog)
    assert event['status']=='SUCCESS' and 'verification' in event['refs']
    handle=event['refs']['verification']
    assert 'sequence_sha256' not in json.dumps(state.artifacts[handle][1])
    event=execute_tool_call({'id':'science1','function':{'name':'investigate_pet_evidence','arguments':json.dumps({
        'query':{'protein_name':'ESM050','ph':6.5,'form':'powder'},'verification_ref':handle})}},state,catalog)
    v=state.results['science1']['result']
    assert event['status']=='PARTIAL' and v['identity']['verified']==1
    assert any(c['category']=='LIMITED_COMPARISON' for c in v['comparisons'])
    assert not v['quantitative_superiority_supported']


@pytest.mark.parametrize('index,expected',[(0,'SUCCESS'),(1,'PARTIAL')])
def test_explicit_measurement_scope_does_not_invent_incomplete_coverage(index,expected):
    question='Inspect multi-study evidence for ESM050 at 40°C, pH 6.5 using PET powder, measurement index 0.'
    scenario=LocalPetScenario({'query':{'protein_name':'ESM050','temperature_c':40,'ph':6.5,'form':'powder','measurement_index':index}})
    report=run_investigation(question,scenario)['result']
    assert ('SUCCESS' if report['status']=='COMPLETE' else 'PARTIAL')==expected
    p=report['trace'][0]['preview']
    assert not p['pazy']['truncated'] and p['pazy']['total_matching_measurements']==1
    assert not p['quantitative_superiority_supported'] and p['identity']['verified']==0
    assert not p['partial_reasons'] and not p['missing_work']
    if index==0:assert not report['validation']['issues']
    else:assert any('Additional source/protocol filters' in t for t in report['validation']['issues'])
    encoded=json.dumps(scenario.inputs)
    assert all(t not in encoded for t in ('amino_acid_sequence','sequence_sha256','"rows"','"vector"','/Users/'))
    assert 'PAZy publication-derived assay:' not in format_research_report(report)


@pytest.mark.parametrize('requested_doi,scope_matches',[('10.1021/acscatal.5c03460',True),
    ('https://doi.org/10.1021/acscatal.5c03460',True),('10.1021/acscatal.5c03460.s002',False)])
def test_exact_doi_scope_retains_metadata_gap_without_fabricating_omissions(requested_doi,scope_matches):
    question=('Inspect multi-study PAZy evidence for ESM050 at 40°C, pH 6.5 using PET powder, '
              f'publication DOI {requested_doi}, measurement index 0.')
    scenario=LocalPetScenario({'query':{'protein_name':'ESM050','publication_doi':'10.1021/acscatal.5c03460',
        'temperature_c':40,'ph':6.5,'form':'powder','measurement_index':0}})
    report=run_investigation(question,scenario)['result']
    p=report['trace'][0]['preview']
    assert report['status']=='PARTIAL' and p['partial_reasons']==['NREL_FILTER_UNVERIFIED']
    assert p['nrel']['total_matches']==p['nrel']['returned']==1
    assert p['pazy']['total_matches']==p['pazy']['returned']==1 and not p['pazy']['truncated']
    assert all(c['publication_doi']=='10.1021/acscatal.5c03460' for c in p['pazy']['records'])
    assert all('omitted' not in text.lower() for text in p['missing_work'])
    assert any('Additional source/protocol filters' in t for t in report['validation']['issues']) is not scope_matches
    facts=[f['text'] for f in report['facts'] if f['text'].startswith('Investigation is PARTIAL:')]
    assert facts and all('omitted' not in text.lower() for text in facts)
