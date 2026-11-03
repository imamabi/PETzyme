"""One offline clarification pathway; session stores context only, never raw turns/results."""
from __future__ import annotations

import streamlit as st
from .clarification import ClarificationTurn, start_clarification, advance_clarification


def clear_clarification() -> None:
    """Flow: Receives external reset/mode-change events from reset_result() or render_clarification(); returns None after clearing active clarification context and input widgets."""
    for key in ('clarification_state', 'clarify_start', 'clarify_reply', 'clarify_kind'):
        st.session_state.pop(key, None)


def _feedback(turn: ClarificationTurn) -> None:
    st.write(turn['message'])
    if turn['triage_outcome'] is not None:
        st.text('triage_protein() outcome: '+turn['triage_outcome']+' · sequence validation: '+str(turn['sequence_validation']))
        st.caption('Transient actual triage outcome. No input sequence, FASTA header or scientific result is retained in session context.')


def submit_start() -> None:
    """Flow: Receives external question str from the clarification form; returns None after start_clarification() stores only ClarificationState and displays transient feedback."""
    question = st.session_state.pop('clarify_start', '')
    try:
        turn = start_clarification(question)
        st.session_state['clarification_state'] = turn['state']
        st.session_state.pop('clarify_reply', None)
        st.session_state['clarify_kind'] = 'ANSWER'
        _feedback(turn)
    except ValueError as error:
        st.error(str(error))


def submit_reply() -> None:
    """Flow: Receives external typed reply str from render_clarification(); returns None after advance_clarification() updates only active ClarificationState and displays transient triage feedback."""
    reply = st.session_state.pop('clarify_reply', '')
    state = st.session_state.get('clarification_state')
    if state is None:
        st.warning('Start a decision context first.')
        return
    turn = advance_clarification(state, reply, st.session_state.get('clarify_kind', 'ANSWER'))
    st.session_state['clarification_state'] = turn['state']
    _feedback(turn)


def render_clarification() -> None:
    """Flow: Receives external widget events from main(); renders active ClarificationState from submit_start()/submit_reply() and transient feedback without calling a reasoning backend."""
    st.subheader('Scientific clarification')
    st.caption('Deterministic local identity triage. No Muse, ESM-2, alignment, network lookup or paid inference.')
    with st.form('clarify_start_form', clear_on_submit=True):
        st.text_area('New two-enzyme decision question', key='clarify_start', max_chars=2000,
            placeholder='Should I use ESM050 or LCCMC? My device supports 40C and 60C, but I cannot operate above pH 7.')
        st.form_submit_button('Start new decision context', key='clarify_begin', on_click=submit_start)
    st.caption('Starting a new context explicitly replaces the previous one. Supported syntax: Should I use NAME or NAME? My device supports temperatures in C/°C; cannot operate above pH N or pH <= N.')
    state = st.session_state.get('clarification_state')
    if state is not None:
        st.text('Original objective: choose between the two requested enzymes under the retained equipment constraints.')
        for i, original in enumerate(state['requested_labels']):
            st.text(f"Enzyme {i+1}: requested {original} · source label {state['confirmed_labels'][i] or 'UNRESOLVED'}")
        st.text('Available temperatures: '+', '.join(f'{t:g}°C' for t in state['temperatures_c'])+
                f" · maximum pH: {state['maximum_ph']:g}")
        st.info(state['pending_question'])
        for original,rejected in zip(state['requested_labels'],state['rejected_labels']):
            if rejected: st.text('Rejected suggestions for '+original+': '+', '.join(rejected))
        st.caption('Source labels and exact sequence matches do not verify experimental constructs or function. No general two-enzyme activity comparison has been run.')
        st.warning('No winning enzyme is established. No experimental conditions outside the retained limits are recommended.')
        if state['pending_index'] is not None:
            with st.form('clarify_reply_form', clear_on_submit=True):
                st.selectbox('Reply type', ['ANSWER','SEQUENCE','REFERENCE'], key='clarify_kind',
                    format_func=lambda k: {'ANSWER':'Confirmation or exact name','SEQUENCE':'Raw sequence / FASTA','REFERENCE':'Accession / DOI'}[k])
                st.text_area('Clarification reply', key='clarify_reply', max_chars=16384)
                st.form_submit_button('Continue clarification', key='clarify_continue', on_click=submit_reply)
            st.caption('Choose Raw sequence / FASTA for raw residues; FASTA headers are never interpreted as identity. Accessions/DOIs are acknowledged without lookup or retention.')
    st.button('Reset scientific clarification', key='clarify_reset', on_click=clear_clarification)
