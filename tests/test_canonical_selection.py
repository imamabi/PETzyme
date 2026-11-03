"""Required-ID contract repair: mocked Meta decisions and real local scientific evidence."""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

import httpx
import pytest

from test_build08b_remediation import QUESTION, TemperatureScenario
from petzyme.agent import run_investigation
from petzyme.agent_execution import execute_tool_call
from petzyme.agent_reporting import (format_research_report, required_source_fact_ids,
                                        validate_investigation)
from petzyme.agent_request import interpret_request, select_tool_catalog
from petzyme.agent_types import AgentState
from petzyme.meta import MAX_BYTES, MetaModelClient
from petzyme.tools import validate_arguments
from petzyme.ui_adapter import ROOT


def call(identifier, arguments):
    return {'id':identifier,'type':'function','function':{
        'name':'investigate_pet_evidence','arguments':json.dumps(arguments)}}


@pytest.fixture(scope='module')
def executed():
    request=interpret_request(QUESTION)
    catalog=select_tool_catalog(request)
    state=AgentState(str(ROOT/'data/nrel/label_data.csv'))
    for temperature in (40,60):
        event=execute_tool_call(call(f't{temperature}',{'query':{
            'protein_name':'ESM050','temperature_c':temperature,'ph':6.5,'form':'powder'}}),state,catalog)
        assert event['status']=='PARTIAL'
    return request,state


def test_required_ids_cover_both_measurements_and_computed_source_assessments(executed):
    request,state=executed
    required=required_source_fact_ids(request,state.events,state.facts)
    assert required==[f't{t}.f{i}' for t in (40,60) for i in range(1,8)]
    chosen=[f for f in state.facts if f['id'] in required]
    assert Counter(f['evidence_category'] for f in chosen)=={'COMPUTED':10,'MEASURED':4}
    assert all(f['provenance'] for f in chosen)
    assert all(f['id'] not in required for f in state.facts if f['kind'] in ('LIMITATION','UNANSWERED'))
    validation=validate_investigation(request,state,{'fact_ids':required})
    assert validation['grounded'] and not validation['complete']
    assert 'Final selection omitted required source evidence facts.' not in validation['issues']
    assert all(f['id'] in validation['selected_fact_ids'] for f in state.facts if f['kind'] in ('LIMITATION','UNANSWERED'))


@pytest.mark.parametrize('missing',range(14))
def test_every_required_fact_remains_mandatory_and_is_not_auto_filled(executed,missing):
    request,state=executed
    required=required_source_fact_ids(request,state.events,state.facts)
    selected=[identifier for i,identifier in enumerate(required) if i!=missing]
    validation=validate_investigation(request,state,{'fact_ids':selected})
    assert validation['grounded'] and not validation['complete']
    assert 'Final selection omitted required source evidence facts.' in validation['issues']
    assert required[missing] not in validation['selected_fact_ids']


@pytest.mark.parametrize('heuristic',['measured_only','latest_only','deduplicated_content','warnings_only'])
def test_plausible_subset_selection_reproduces_live_failure_without_claiming_live_ids(executed,heuristic):
    request,state=executed
    required=required_source_fact_ids(request,state.events,state.facts)
    if heuristic=='measured_only': selected=[f['id'] for f in state.facts if f['evidence_category']=='MEASURED']
    elif heuristic=='latest_only': selected=[f for f in required if f.startswith('t60.')]
    elif heuristic=='warnings_only': selected=[f['id'] for f in state.facts if f['kind'] in ('LIMITATION','UNANSWERED')]
    else:
        texts=set();selected=[]
        for fact in state.facts:
            if fact['id'] in required and fact['text'] not in texts:
                texts.add(fact['text']);selected.append(fact['id'])
        assert len(required)-len(selected)==1
    result=validate_investigation(request,state,{'fact_ids':selected})
    assert result['grounded'] and 'Final selection omitted required source evidence facts.' in result['issues']


class ChecklistScenario(TemperatureScenario):
    def __init__(self, mode='complete'):
        """Flow: Receives external offline scenario mode str; initializes mocked responses for respond()."""
        super().__init__();self.mode=mode

    def respond(self, request):
        """Flow: Receives httpx.Request from MetaModelClient.complete() mock transport; returns synthetic httpx.Response using its advertised final-selection checklist."""
        response=super().respond(request).json()
        if response['choices'][0]['finish_reason']=='stop':
            payload=json.loads(request.content)
            replies=[json.loads(m['content']) for m in payload['messages'] if m['role']=='tool']
            required=replies[-1]['final_selection']['required_fact_ids']
            if self.mode=='drop_one':required=required[:-1]
            elif self.mode=='drop_aliases':
                aliases={f['id'] for r in replies for f in r['facts'] if 'same_as' in f}
                required=[f for f in required if f not in aliases]
            content=json.dumps({'fact_ids':required})
            if self.mode=='unsupported':content='Temperature causes superior activity and verified kinetic rates.'
            if self.mode=='invented':content=json.dumps({'fact_ids':required+['invented.f999']})
            response['choices'][0]['message']['content']=content
        return httpx.Response(200,json=response)


@pytest.mark.parametrize('mode',['complete','drop_one','drop_aliases','unsupported','invented'])
def test_advertised_contract_integrates_without_weakening_science_or_grounding(mode):
    scenario=ChecklistScenario(mode)
    report=run_investigation(QUESTION,MetaModelClient(transport=httpx.MockTransport(scenario.respond)))['result']
    assert report['model_requests']==3 and report['tool_attempts']==2 and report['status']=='PARTIAL'
    replies=[json.loads(m['content']) for m in scenario.requests[-1]['messages'] if m['role']=='tool']
    first,second=[r['final_selection'] for r in replies]
    assert len(first['required_fact_ids'])==7 and len(second['required_fact_ids'])==14
    assert set(first['required_fact_ids'])<set(second['required_fact_ids'])
    assert second['automatically_retained_kinds']==['LIMITATION','UNANSWERED'] and second['max_selected_ids']==100
    aliases={f['id'] for r in replies for f in r['facts'] if 'same_as' in f and f['kind']=='EVIDENCE'}
    assert len(aliases)==1 and aliases <= set(second['required_fact_ids'])
    assert report['synthesis']['status']==('VALIDATED_FACT_SELECTION' if mode=='complete' else 'INCOMPLETE')
    if mode=='complete':
        assert report['validation']['grounded'] and not report['validation']['complete']
        assert all(f in report['validation']['selected_fact_ids'] for f in second['required_fact_ids'])
    if mode in ('drop_one','drop_aliases'):
        assert report['validation']['grounded']
        assert 'Final selection omitted required source evidence facts.' in report['validation']['issues']
    if mode in ('unsupported','invented'):assert not report['validation']['grounded']
    observations=[o for e in report['trace'] for o in e['preview']['nrel']['observations']]
    assert [o['original_text'] for o in observations]==['0.08499999999999999','0.13']
    assert all(o['unit_status']=='UNKNOWN/UNVERIFIED' and o['ph']==6.5 and o['pet_form']=='CRYSTALLINE_POWDER' for o in observations)
    assert all(not e['preview']['quantitative_superiority_supported'] and not e['preview']['causal_claim_supported'] for e in report['trace'])
    assert 'PAZy publication-derived assay:' not in format_research_report(report)
    assert 'Temperature causes superior' not in format_research_report(report)
    outbound=json.dumps(scenario.requests)
    assert all(token not in outbound for token in ('amino_acid_sequence','sequence_sha256','"rows"','"vector"','/Users/'))
    assert all(len(json.dumps(r).encode())<MAX_BYTES for r in scenario.requests)


@pytest.mark.parametrize('limit',[0,4,10])
def test_undocumented_limit_bounds_reproduced_and_now_advertised_before_execution(limit,monkeypatch):
    catalog=select_tool_catalog(interpret_request(QUESTION))
    schema=catalog[0]['function']['parameters']
    old=deepcopy(schema)
    old['properties']['limit'].pop('minimum');old['properties']['limit'].pop('maximum')
    arguments={'query':{'protein_name':'ESM050'},'limit':limit}
    # Original generated schema accepted this, but investigate_pet_evidence() already rejected it.
    assert validate_arguments(arguments,old)
    from petzyme.pet_evidence import investigate_pet_evidence
    assert investigate_pet_evidence(**arguments)['status']=='ERROR'
    assert not validate_arguments(arguments,schema)
    monkeypatch.setattr('petzyme.agent_execution.invoke_tool',lambda *a,**k:pytest.fail('invalid limit must fail before invocation'))
    state=AgentState(str(ROOT/'data/nrel/label_data.csv'))
    event=execute_tool_call(call('invalid_limit',arguments),state,catalog)
    assert event['status']=='ERROR' and not required_source_fact_ids(interpret_request(QUESTION),state.events,state.facts)


@pytest.mark.parametrize('arguments',[
    {'query':{'enzyme_identifier':'ESM050'}},
    {'query':{'temperature_c':[40,60]}},
    {'query':{'temperature_c':None}},
    {'verification_ref':None},
])
def test_invalid_shapes_remain_rejected_not_coerced(arguments):
    catalog=select_tool_catalog(interpret_request(QUESTION))
    assert not validate_arguments(arguments,catalog[0]['function']['parameters'])
    state=AgentState(str(ROOT/'data/nrel/label_data.csv'))
    assert execute_tool_call(call('bad_shape',arguments),state,catalog)['status']=='ERROR'


@pytest.mark.parametrize('limit',[1,3])
def test_valid_limit_boundaries_remain_usable(limit):
    catalog=select_tool_catalog(interpret_request(QUESTION))
    args={'query':{'protein_name':'ESM050','temperature_c':40,'ph':6.5,'form':'powder'},'limit':limit}
    assert validate_arguments(args,catalog[0]['function']['parameters'])
    state=AgentState(str(ROOT/'data/nrel/label_data.csv'))
    assert execute_tool_call(call('valid_limit',args),state,catalog)['status']=='PARTIAL'


@pytest.mark.parametrize('mode,selected,omitted',[('complete',14,0),('drop_one',13,1),('drop_aliases',13,1),('invented',0,14)])
def test_safe_count_diagnostics_describe_actual_selection_without_ids(mode,selected,omitted):
    scenario=ChecklistScenario(mode)
    report=run_investigation(QUESTION,MetaModelClient(transport=httpx.MockTransport(scenario.respond)))['result']
    counts=report['validation']['selection_counts']
    assert counts=={'required':14,'selected':selected,'omitted':omitted}
    assert all(type(value) is int for value in counts.values())
