from copy import deepcopy
import json
import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from test_esm import SyntheticRuntime
from zymecycle import esm, ui_live, ui_adapter

APP = Path(__file__).resolve().parents[1] / 'tests/legacy_streamlit_app.py'


def app():
    return AppTest.from_file(APP, default_timeout=30).run()


def test_initial_launch_examples_and_reruns_execute_nothing(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected investigation or Keychain access')
    monkeypatch.setattr(ui_adapter, 'run_offline', forbidden)
    monkeypatch.setattr(ui_live.subprocess, 'Popen', forbidden)
    at = app()
    assert not at.exception
    assert at.sidebar.radio[0].value == 'Offline demonstration'
    assert 'view' not in at.session_state
    for i in range(4):
        at.button(key=f'example_{i}').click().run()
        assert at.text_area(key='question').value == list(ui_adapter.EXAMPLES.values())[i]
        assert not at.exception and 'view' not in at.session_state
    at.run()
    assert not at.exception


@pytest.mark.parametrize('index', [0,1,2,3])
def test_offline_examples_render_real_tool_results(index, monkeypatch):
    monkeypatch.setattr(esm,'load_esm2_runtime',lambda:SyntheticRuntime())
    at = app()
    at.button(key=f'example_{index}').click().run()
    at.button(key='investigate').click().run()
    assert not at.exception, at.exception
    view = at.session_state['view']
    assert view['workflow_status']=='COMPLETE' and view['mode']=='DETERMINISTIC_OFFLINE'
    assert at.metric[0].value=='COMPLETE'
    assert any('not establish sufficient scientific evidence' in c.value for c in at.caption)
    assert any('Scientific limitations' in m.value for m in at.markdown)
    if index==0:
        table = at.dataframe[0].value
        assert table['Recorded response'].tolist()==['0.085','0.13']
        assert view['observations'][0]['original_text']=='0.08499999999999999'
        assert 'unsupported' in at.info[0].value
    if index==1:
        assert view['observations']==[]
        assert at.dataframe[0].value['Observation'].tolist()==['MISSING — not zero','MISSING — not zero']
    if index==2:
        assert view['similarity']['coverage']['searched_candidates']==8
        assert any('entire dataset was not searched' in w.value for w in at.warning)
    if index==3:
        actions=view['plan']['recommendations']
        assert len(actions)==7 and actions[-1]['prerequisites']
        assert all(a['evidence_category']=='HYPOTHETICAL' for a in actions)
        assert any('PROPOSAL / HYPOTHETICAL' in m.value for m in at.markdown)


def test_unsupported_question_and_tool_failure_are_not_fabricated(monkeypatch):
    at=app()
    at.text_area(key='question').input('Invent a prediction for a new enzyme')
    at.button(key='investigate').click().run()
    assert not at.exception and at.error and 'Prediction and simulation are unsupported' in at.error[0].value
    def failure(*args,**kwargs): raise RuntimeError('PRIVATE_ERROR')
    monkeypatch.setattr(ui_adapter,'run_offline',failure)
    at.button(key='example_0').click().run()
    at.button(key='investigate').click().run()
    assert not at.exception and 'No substitute evidence' in at.error[0].value
    assert 'PRIVATE_ERROR' not in at.error[0].value


def test_live_mode_never_launches_without_authorized_submission(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('No launch authorized')
    monkeypatch.setattr(ui_live.subprocess,'Popen',forbidden)
    at=app()
    at.sidebar.radio[0].set_value('Live Meta agent').run()
    at.button(key='example_0').click().run()
    at.run()
    assert not at.exception
    at.button(key='investigate').click().run()
    assert not at.exception and 'Authorize live' in at.error[0].value


def test_per_investigation_authorization_consumed_and_reruns_do_not_repeat(monkeypatch):
    safe=ui_adapter.run_offline(ui_adapter.EXAMPLES['Temperature evidence'])
    live=deepcopy(safe)
    live['result']['mode']='MOCK_META'
    calls=[]
    original=ui_live.run_keychain_investigation
    def mock_live(question,*,authorized=False,on_update=None,completion_tokens=2048,runtime_limits=None,cost_ceiling_usd=None):
        if not authorized: return original(question,authorized=False)
        calls.append(question)
        return deepcopy(live)
    monkeypatch.setattr(ui_live,'run_keychain_investigation',mock_live)
    at=app()
    at.sidebar.radio[0].set_value('Live Meta agent').run()
    at.checkbox[0].check()
    at.button(key='investigate').click().run()
    assert len(calls)==1 and not at.exception
    assert at.checkbox[0].value is False
    at.run()
    assert len(calls)==1
    at.button(key='investigate').click().run()
    assert len(calls)==1 and 'Authorize live' in at.error[0].value


def test_partial_workflow_distinct_and_limitations_visible(monkeypatch):
    def partial(question,**kwargs):
        return ui_adapter.run_offline_saved(question,csv_path='missing.csv')
    monkeypatch.setattr(ui_adapter,'run_offline_saved',ui_adapter.run_offline,raising=False)
    monkeypatch.setattr(ui_adapter,'run_offline',partial)
    at=app()
    at.button(key='investigate').click().run()
    assert not at.exception and at.metric[0].value=='PARTIAL'
    assert at.warning and at.session_state['view']['observations']==[]


def test_public_ui_keeps_pazy_records_and_provider_details_out():
    from zymecycle.results import outcome
    from zymecycle.agent_evidence import public_tool_update
    native=outcome('SUCCESS',{'id':987654,'name':'PRIVATE_RECORD','sequence':'ACDEF'*20},[],[],None)
    update=public_tool_update({'call_id':'p1','name':'search_pazy_pet','status':'SUCCESS','ephemeral':True},native,1,1)
    report=outcome('PARTIAL',{'question':'PRIVATE_RECORD 987654','provider_mode':'MOCK_META','status':'PARTIAL',
        'model_requests':1,'tool_attempts':1,'validation':{'issues':['Partial retrieval']},
        'facts':[{'text':'PRIVATE_PROVIDER_REASONING'}],'limitations':[],
        'persistence':'FORBIDDEN_PAZY_DERIVED'},[],[],'COMPUTED')
    public=ui_adapter.present_investigation(report,[update])['result']
    at=app()
    at.session_state['view']=public
    at.run()
    assert not at.exception
    text=json.dumps(at.session_state['view'])+' '.join(m.value for m in at.markdown)
    assert 'PRIVATE_' not in text and '987654' not in text and 'ACDEF'*20 not in text
    assert 'withheld' in text


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get('ZYMECYCLE_REAL_ESM')!='1',reason='Real local inference opt-in required')
def test_apptest_real_pretrained_similarity():
    at=app()
    at.button(key='example_2').click().run()
    at.button(key='investigate').click().run()
    assert not at.exception
    similarity=at.session_state['view']['similarity']
    assert similarity['model']['checkpoint']=='facebook/esm2_t12_35M_UR50D'
    assert similarity['coverage']['searched_candidates']==8
    assert similarity['coverage']['outside_window_records']==204
    assert at.dataframe[0].value['Protein'].tolist()[0]=='DP038'
