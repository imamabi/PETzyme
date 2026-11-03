"""Essential offline configuration, transport, actual process-boundary and UI checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from test_build08b_remediation import TemperatureScenario, QUESTION
from zymecycle.meta import (COMPLETION_ALLOWANCES, MAX_COMPLETION_TOKENS, MAX_BYTES,
    REQUEST_TIMEOUT_SECONDS, MetaModelClient, ModelError)
from zymecycle.meta_telemetry import estimated_token_cost
from zymecycle.ui_adapter import ROOT, investigate_with_backend


def transport_for(scenario, allowance):
    def respond(request):
        body=json.loads(request.content)
        assert type(body['max_completion_tokens']) is int and body['max_completion_tokens']==allowance
        assert 'reasoning_effort' not in body and 'reasoning_strength' not in body
        assert request.extensions['timeout']['read']==REQUEST_TIMEOUT_SECONDS==180
        response=scenario.respond(request).json()
        response['usage']={'prompt_tokens':1000,'completion_tokens':100,'total_tokens':1100,
            'prompt_tokens_details':{'cached_tokens':100},'completion_tokens_details':{'reasoning_tokens':80}}
        response['private_provider_field']='NEVER_PERSIST_PROVIDER_BODY'
        return httpx.Response(200,json=response)
    return httpx.MockTransport(respond)


@pytest.mark.parametrize('allowance',COMPLETION_ALLOWANCES)
def test_all_allowances_real_science_grounding_privacy_and_telemetry(allowance):
    scenario=TemperatureScenario(temperatures=(40,60,40))
    client=MetaModelClient(transport=transport_for(scenario,allowance),completion_tokens=allowance)
    view=investigate_with_backend(QUESTION,client)['result']
    assert view['synthesis']['status']=='VALIDATED_FACT_SELECTION' and view['workflow_status']=='PARTIAL'
    union=view['multi_study']['accumulation']
    assert union['condition_coverage']['complete'] and union['unique_nrel_retrieved']==2
    assert union['unique_pazy_assays_retrieved']==2
    assert [o['original_text'] for o in view['multi_study']['nrel']['observations']]==['0.08499999999999999','0.13']
    entries=view['provider_telemetry']
    assert len(entries)==4 and len(scenario.requests)==4
    assert all(e['http_status']==200 and e['reasoning_tokens']==80 and e['visible_output_tokens']==20 for e in entries)
    assert all(e['request_bytes']==len(json.dumps(r,separators=(',',':'),ensure_ascii=False).encode()) for e,r in zip(entries,scenario.requests))
    assert estimated_token_cost(entries)==pytest.approx(4*(900*1.25+100*.15+100*4.25)/1e6)
    assert 'NEVER_PERSIST' not in json.dumps(view)
    assert all(s not in json.dumps(scenario.requests) for s in ('amino_acid_sequence','sequence_sha256','"rows"','"vector"','/Users/'))


@pytest.mark.parametrize('bad',[True,15000,200000,0,'10000',None])
def test_invalid_allowance_cannot_start_provider_or_worker(bad,monkeypatch):
    from zymecycle import ui_live
    with pytest.raises(ValueError):MetaModelClient(completion_tokens=bad)
    monkeypatch.setattr(ui_live.subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch'))
    assert ui_live.run_keychain_investigation(QUESTION,authorized=True,completion_tokens=bad)['status']=='ERROR'


@pytest.mark.parametrize('finish',['length','content_filter','UNKNOWN_PRIVATE_FINISH'])
def test_incomplete_rejected_without_raw_response_retention(finish):
    scenario=TemperatureScenario(finish=finish,final='PRIVATE_PARTIAL_TEXT')
    client=MetaModelClient(transport=transport_for(scenario,10000),completion_tokens=10000)
    view=investigate_with_backend(QUESTION,client)['result']
    assert view['synthesis']['status']=='INCOMPLETE' and len(client.telemetry)==3
    assert all(v not in json.dumps(view) for v in ('PRIVATE_PARTIAL_TEXT','UNKNOWN_PRIVATE_FINISH','NEVER_PERSIST'))
    assert client.telemetry[-1]['error_code'] is not None


def test_missing_usage_timeout_http_and_original_size_bounds():
    client=MetaModelClient(transport=httpx.MockTransport(lambda r:httpx.Response(401,text='SECRET_BODY')))
    with pytest.raises(ModelError):client.complete([],[])
    assert client.telemetry[0]['http_status']==401 and estimated_token_cost(client.telemetry) is None
    assert 'SECRET_BODY' not in json.dumps(client.telemetry)
    def timeout(request):raise httpx.ReadTimeout('PRIVATE_TIMEOUT',request=request)
    client=MetaModelClient(transport=httpx.MockTransport(timeout))
    with pytest.raises(ModelError):client.complete([],[])
    assert client.telemetry[0]['error_code']=='TIMEOUT' and client.telemetry[0]['http_status'] is None
    client=MetaModelClient(transport=httpx.MockTransport(lambda r:pytest.fail('no HTTP')))
    with pytest.raises(ModelError):client.complete([{'role':'user','content':'x'*MAX_BYTES}],[])
    assert not client.telemetry and MAX_BYTES==262144


def test_original_default_and_no_client_lifetime_budget():
    scenario=TemperatureScenario()
    client=MetaModelClient(transport=httpx.MockTransport(scenario.respond))
    for _ in range(3):
        view=investigate_with_backend(QUESTION,client)['result']
        assert view['synthesis']['status']=='VALIDATED_FACT_SELECTION'
        assert len(view['provider_telemetry'])==3
    assert len(scenario.requests)==9 and MAX_COMPLETION_TOKENS==2048
    assert all(r['max_completion_tokens']==2048 for r in scenario.requests)


@pytest.mark.parametrize('allowance',COMPLETION_ALLOWANCES)
def test_exact_parent_launcher_worker_command_with_offline_dependencies(allowance,tmp_path,monkeypatch):
    from zymecycle import ui_live
    # Test-only module wrappers substitute the two external dependencies, not execution/argument parsing.
    # PYTHONSAFEPATH makes the wrappers precede cwd; the production -m commands remain exact.
    for package in ('scripts','zymecycle'):
        directory=tmp_path/package;directory.mkdir()
        (directory/'__init__.py').write_text('__path__.append('+repr(str(ROOT/package))+')\n')
    common='''import importlib.util,socket,sys,os
from pathlib import Path
ROOT=Path(%r)
def blocked(*a,**k):raise AssertionError('Offline child must never connect')
socket.socket.connect=blocked
''' % str(ROOT)
    (tmp_path/'scripts/launch_meta.py').write_text(common+'''
import subprocess
spec=importlib.util.spec_from_file_location('scripts._real_launch_meta',ROOT/'scripts/launch_meta.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert os.environ.get('MODEL_API_KEY') == 'synthetic-parent-must-be-removed'
module._lookup=lambda password:subprocess.CompletedProcess([],0,b'synthetic-child-only\\n')
raise SystemExit(module.main())
''')
    (tmp_path/'zymecycle/ui_live.py').write_text(common+'''
import httpx
from tests.test_build08b_remediation import TemperatureScenario
from zymecycle import ui_adapter,meta
assert os.environ.get('MODEL_API_KEY')=='synthetic-child-only'
scenario=TemperatureScenario()
Original=meta.MetaModelClient
meta.MetaModelClient=lambda **kwargs:Original(transport=httpx.MockTransport(scenario.respond),completion_tokens=kwargs['completion_tokens'],max_estimated_cost_usd=kwargs.get('max_estimated_cost_usd'))
spec=importlib.util.spec_from_file_location('zymecycle._real_ui_live',ROOT/'zymecycle/ui_live.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
raise SystemExit(module.worker_main())
''')
    monkeypatch.setenv('ZYMECYCLE_META_COST_CEILING_USD','1')
    monkeypatch.setenv('PYTHONSAFEPATH','1')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path)+os.pathsep+str(ROOT))
    monkeypatch.setenv('MODEL_API_KEY','synthetic-parent-must-be-removed')
    popen=ui_live.subprocess.Popen;commands=[]
    def observe(command,**kwargs):
        commands.append(command)
        return popen(command,**kwargs)
    monkeypatch.setattr(ui_live.subprocess,'Popen',observe)
    updates=[]
    result=ui_live.run_keychain_investigation(QUESTION,authorized=True,completion_tokens=allowance,on_update=updates.append)
    assert len(commands)==1 and commands[0]==[sys.executable,'-m','scripts.launch_meta','--authorize-paid-meta','--',sys.executable,'-m','zymecycle.ui_live','--live-worker']
    assert result['result'] is not None and len(updates)==2
    view=result['result']
    assert view['synthesis']['status']=='VALIDATED_FACT_SELECTION'
    assert view['multi_study']['accumulation']['condition_coverage']['complete']
    assert len(view['provider_telemetry'])==3
    assert view['provider_budget']=={'limit_usd':1.0,'blocked':False}
    assert view['selection_counts']=={'required':14,'selected':14,'omitted':0}
    assert all(e['completion_allowance']==allowance for e in view['provider_telemetry'])
    assert 'synthetic-child-only' not in json.dumps(result) and 'synthetic-parent-must-be-removed' not in json.dumps(result)
    assert not list(tmp_path.rglob('*.json'))


@pytest.mark.parametrize('finish',['stop','length'])
def test_minimal_live_ui_controls_with_mocked_provider(monkeypatch,finish):
    from zymecycle import ui_live
    calls=[]
    def fake_live(question,*,authorized,on_update,completion_tokens,runtime_limits,cost_ceiling_usd):
        calls.append(completion_tokens);assert authorized
        scenario=TemperatureScenario(finish=finish)
        return investigate_with_backend(question,MetaModelClient(transport=transport_for(scenario,completion_tokens),
            completion_tokens=completion_tokens),on_update=on_update)
    monkeypatch.setattr(ui_live,'run_keychain_investigation',fake_live)
    at=AppTest.from_file(ROOT/'tests/legacy_streamlit_app.py',default_timeout=30).run()
    at.radio(key='mode').set_value('Live Meta agent').run()
    assert at.selectbox(key='completion_allowance').value==10000
    at.text_area(key='question').input(QUESTION)
    at.selectbox(key='completion_allowance').set_value(25000)
    at.checkbox[0].check();at.button(key='investigate').click().run()
    assert not at.exception and calls==[25000]
    assert 'view' not in at.session_state and not at.json and not at.dataframe and not at.get('download_button')
    assert any(e.value=='Live Muse synthesis incomplete.' for e in at.error)==(finish=='length')
    text=' '.join(t.value for t in at.text)
    assert '0.08499999999999999' in text and '0.13' in text and 'UNKNOWN/UNVERIFIED' in text
    at.run()
    assert calls==[25000] and not at.text and 'view' not in at.session_state


@pytest.mark.parametrize('missing_usage',[False,True])
def test_spending_guard_blocks_before_an_extra_request(missing_usage):
    calls=[]
    def respond(request):
        calls.append(1)
        body={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'{}'}}]}
        if not missing_usage:body['usage']={'prompt_tokens':10000,'completion_tokens':1000,'total_tokens':11000}
        return httpx.Response(200,json=body)
    client=MetaModelClient(transport=httpx.MockTransport(respond),completion_tokens=10000,max_estimated_cost_usd=.05)
    client.complete([],[])
    with pytest.raises(ModelError,match='spending ceiling'):client.complete([],[])
    assert len(calls)==len(client.telemetry)==1 and client.budget_blocked


@pytest.mark.parametrize('invalid',[0,-1,True,float('inf'),float('nan'),'1'])
def test_invalid_spending_ceiling_never_launches(invalid):
    with pytest.raises(ValueError):MetaModelClient(max_estimated_cost_usd=invalid)
