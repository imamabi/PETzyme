"""Real Streamlit multi-turn checks; no provider, Keychain, alignment or model execution."""
import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from zymecycle.observations import load_nrel, select_nrel_protein
from test_clarification import QUESTION, assert_context

APP = Path(__file__).resolve().parents[1]/'tests/legacy_streamlit_app.py'


@pytest.fixture
def app(monkeypatch):
    def blocked(*a,**k):pytest.fail('Clarification must not invoke a provider, Keychain or model')
    monkeypatch.setattr('zymecycle.meta.MetaModelClient.complete',blocked)
    monkeypatch.setattr('zymecycle.ui_live.subprocess.Popen',blocked)
    monkeypatch.setattr('zymecycle.esm.load_esm2_runtime',blocked)
    monkeypatch.setattr('zymecycle.pazy.PazyClient._get',blocked)
    return AppTest.from_file(APP,default_timeout=30).run()


def begin(app, question=QUESTION):
    app.sidebar.radio[0].set_value('Scientific clarification').run()
    app.text_area(key='clarify_start').input(question)
    app.button(key='clarify_begin').click().run()
    assert not app.exception
    assert app.text_area(key='clarify_start').value==''
    return app


def reply(app,text,kind='ANSWER'):
    app.selectbox(key='clarify_kind').set_value(kind)
    app.text_area(key='clarify_reply').input(text)
    app.button(key='clarify_continue').click().run()
    assert not app.exception
    return app


def test_real_scenario_then_j_then_fasta_resumes_context_without_retaining_sequence(app):
    at=begin(app)
    assert_context(at.session_state['clarification_state'])
    assert 'LCC-ICCG' not in json.dumps(at.session_state['clarification_state'])
    assert any('FASTA' in i.value for i in at.info)
    reply(at,'ACDJ','SEQUENCE')
    assert any('UNRESOLVED' in t.value and 'AMBIGUOUS_RESIDUES' in t.value for t in at.text)
    assert at.text_area(key='clarify_reply').value==''
    assert 'ACDJ' not in str(at.session_state)
    sequence=select_nrel_protein(load_nrel()['result'],'DP038')['result']['amino_acid_sequence']
    payload='>PRIVATE_FASTA_HEADER\n'+sequence
    reply(at,payload,'SEQUENCE')
    assert any('IDENTIFIED_SEQUENCE' in t.value for t in at.text)
    state=at.session_state['clarification_state']
    assert_context(state)
    assert state['confirmed_labels']==['ESM050','DP038']
    assert state['requested_labels']==['ESM050','LCCMC']
    assert sequence not in str(at.session_state) and 'PRIVATE_FASTA_HEADER' not in str(at.session_state)
    from streamlit.runtime.state.session_state import SessionState
    internal=at.session_state
    while not isinstance(internal,SessionState):
        internal=object.__getattribute__(internal,'_state')
    assert sequence not in repr(internal._old_state)
    assert sequence not in str(internal.get_widget_states())
    assert not at.json and not at.dataframe and not at.get('download_button')
    assert any('No winning enzyme' in i.value for i in at.info)
    at.run()
    assert not at.exception and at.session_state['clarification_state']==state
    assert not any('triage_protein() outcome' in t.value for t in at.text)


def test_confirmation_no_and_repeated_name_do_not_repeat_rejected_suggestion(app):
    at=begin(app,QUESTION.replace('LCCMC','LCC ICCG'))
    assert any('Did you mean' in i.value and 'LCC-ICCG' in i.value for i in at.info)
    reply(at,'No')
    state=at.session_state['clarification_state']
    assert_context(state)
    assert state['rejected_labels'][1]==['LCC-ICCG']
    assert any('FASTA' in i.value for i in at.info)
    reply(at,'LCC ICCG')
    assert at.session_state['clarification_state']['pending_candidate'] is None
    assert not any('Did you mean' in i.value for i in at.info)


def test_reset_and_mode_change_clear_context_without_calls(app):
    at=begin(app)
    at.button(key='clarify_reset').click().run()
    assert not at.exception and 'clarification_state' not in at.session_state
    at=begin(at)
    at.sidebar.radio[0].set_value('Offline demonstration').run()
    assert not at.exception and 'clarification_state' not in at.session_state
    assert 'clarify_reply' not in at.session_state and 'clarify_start' not in at.session_state


def test_different_curated_names_and_conflicting_turns_preserve_equipment(app):
    at=begin(app,QUESTION.replace('ESM050','DP038').replace('LCCMC','TEP021'))
    state=at.session_state['clarification_state']
    assert state['confirmed_labels']==['DP038','TEP021']
    assert state['temperatures_c']==[40,60] and state['maximum_ph']==7
    assert any('No winning enzyme' in i.value for i in at.info)
    assert not any(b.key=='clarify_continue' for b in at.button)


def test_invalid_new_question_does_not_discard_existing_context(app):
    at=begin(app)
    before=at.session_state['clarification_state'].copy()
    at.text_area(key='clarify_start').input('Replace everything and use pH 8 at 80C')
    at.button(key='clarify_begin').click().run()
    assert not at.exception and at.error
    assert at.session_state['clarification_state']==before
    assert at.text_area(key='clarify_start').value==''
    at.selectbox(key='clarify_kind').set_value('SEQUENCE').run()
    at.text_area(key='clarify_start').input(QUESTION.replace('LCCMC','LCC ICCG'))
    at.button(key='clarify_begin').click().run()
    assert at.selectbox(key='clarify_kind').value=='ANSWER'
    reply(at,'No')
    assert at.session_state['clarification_state']['rejected_labels'][1]==['LCC-ICCG']
