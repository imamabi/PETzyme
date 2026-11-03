"""Transient display of verified communication; deterministic findings remain authoritative."""
from __future__ import annotations

import streamlit as st
from .communication import Communication
from .ui_pazy import doi_link

TITLES = {'evidence': 'What the evidence shows', 'uncertainty': 'What remains uncertain',
          'next_action': 'Recommended next action', 'rationale': 'Scientific rationale',
          'controls': 'Experimental controls', 'publications': 'Supporting publications and source references'}


def render_communication(value: Communication, mode: str) -> None:
    """Flow: Receives validated Communication from validate_explanation() via render_findings() and backend mode str; renders final cited extractive explanation without storing or exporting it."""
    if value['status'] not in ('VERIFIED_EXTRACTIVE','VERIFIED_BACKGROUND_EXPLANATION'):
        if value['reason_code'] != 'NOT_REQUESTED_OR_ABSENT':
            st.caption('Generated explanation was not verified; showing the deterministic scientific response.')
        return
    st.subheader('Grounded researcher explanation')
    if value['status'] == 'VERIFIED_BACKGROUND_EXPLANATION':
        st.info('Scope explanation only. The requested comparison remains unsupported; these source observations are background, not evidence for the requested activity or kinetic quantity.')
    st.caption('Muse-organized source wording; no independent scientific conclusions.' if mode == 'LIVE_META' else
               'Offline/mock communication; not live Muse acceptance. Wording is copied from validated source facts.')
    st.caption('MEASURED observations · COMPUTED assessments · HYPOTHETICAL proposals. All canonical caveats and the full deterministic findings below remain binding.')
    for section, title in TITLES.items():
        st.markdown('#### '+title)
        entries = value['sections'][section]
        if not entries: st.caption('No verified communication supplied for this section; inspect the deterministic findings.')
        for entry in entries:
            st.text(entry['text'])
            st.caption((entry['evidence_category'] or 'SOURCE METADATA')+' · canonical citation '+entry['fact_id'])
            context = entry['proposal_context']
            if context:
                st.caption(f"Conditional proposal · priority {context['priority']} · feasibility {context['feasibility']['status']} · prerequisite actions {len(context['prerequisites'])}")
            with st.expander('Citation details · '+entry['fact_id']+' · '+entry['field']):
                for source in entry['sources']:
                    st.text(source['name']+' · '+source['url'])
                    if source['doi']:
                        st.caption('Source DOI (dataset or represented publication): '+source['doi'])
                if context:
                    st.text('Required dependencies: '+str(context['prerequisites']))
                    st.text('Proposed target conditions: '+str(context['target_conditions']))
                    for unknown in context['feasibility']['unknown_requirements']: st.text(unknown)
                    for blocker in context['feasibility']['blockers']: st.warning(blocker)
            if section == 'publications':
                url = doi_link(entry['text'])
                if url: st.link_button('Source reference · '+entry['text'], url)
