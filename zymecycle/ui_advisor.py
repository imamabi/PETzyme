"""Concise transient experiment-advice presentation; no cache, export or session writes."""
from __future__ import annotations

import streamlit as st
from .runtime_config import DEFAULT_TURN_MODELS, DEFAULT_TURN_TOOLS
from .ui_adapter import ResearchView
from .ui_findings import display_response, prioritize_limitations
from .ui_multistudy import render_multi_study
from .ui_pazy import doi_link


def render_research_advice(view: ResearchView) -> None:
    """Flow: Receives transient ResearchView from present_investigation() via render_findings(); renders final evidence, hypothetical actions, controls and source-linked publications without persistence."""
    st.divider()
    st.subheader('Evidence → next experiment')
    st.caption('Deterministic scientific assessment · COMPUTED. Proposals · HYPOTHETICAL. No experiment has been performed.')
    responses = '; '.join(f"NREL records {display_response(o['original_text'])} at {o['temperature_c']}°C (units {o['unit_status']})"
                          for o in view['observations'])
    st.info((responses+'. ' if responses else '')+view['advice']['answer'])
    plan = view['plan']
    actions = plan['recommendations']
    if actions:
        first = actions[0]
        st.markdown('#### Recommended next action · HYPOTHETICAL')
        st.write(first['proposed_action'])
        st.caption(f"Feasibility {first['feasibility']['status']} · no laboratory resources, costs or outcomes inferred.")
        st.markdown('#### Scientific rationale · COMPUTED')
        st.write(first['expected_information'])
        st.write('Source verification precedes dependent measurements. The advisor checks PAZy coverage before proposing measurements: documented targets require a verification or replication rationale. PAZy does not authenticate identity or add independent replication.')
    st.markdown('#### Proposed experiment and controls · HYPOTHETICAL')
    st.caption(f"{len(actions)} of {plan['candidate_count']} actions shown; {plan['omitted_count']} omitted. Priorities are ordinal, not estimated information gain.")
    priorities = {a['action_id']: str(a['priority']) for a in actions}
    for action in actions:
        with st.expander(f"{action['priority']}. {action['code'].replace('_', ' ').capitalize()}",
                         expanded=action['code'] == 'CONTROLLED_COMPARISON'):
            st.write(action['proposed_action'])
            st.write('Uncertainty addressed: '+action['expected_information'])
            st.caption(action['category']+' · HYPOTHETICAL · feasibility '+action['feasibility']['status'])
            st.write('Prerequisites: '+(', '.join(priorities.get(p, p) for p in action['prerequisites']) or 'None'))
            for requirement in action['required_measurements']: st.write('• '+requirement)
            st.write('Controlled: '+('; '.join(action['controlled_variables']) or 'Source verification; no assay specified.'))
            st.write('Varied: '+('; '.join(action['varied_variables']) or 'None specified.'))
            st.write('Target conditions: '+str(action['target_conditions']))
            st.write('Unknown requirements: '+'; '.join(action['feasibility']['unknown_requirements']))
            for blocker in action['feasibility']['blockers']: st.warning(blocker)
            st.caption(action['ranking_rationale'])
    st.markdown('#### Evidence and provenance · MEASURED observations')
    for o in view['observations']:
        st.text(f"NREL {o['enzyme_identifier']} · {o['temperature_c']}°C · pH {o['ph']} · {o['pet_form']} · recorded response {display_response(o['original_text'])} · units {o['unit_status']}")
    with st.expander('Exact NREL values and recommendation evidence'):
        for o in view['observations']:
            st.text(f"{o['source_column']}: {o['original_text']} · record {o['source_row']} · CSV SHA-256 {o['csv_sha256']}")
        for action in actions:
            for basis in action['evidence_basis']:
                st.text(f"{action['priority']} · {basis['category']} · {basis['statement']}")
                for source in basis['sources']: st.text(source['url']+' · '+source['locator'])
    if view.get('multi_study'):
        with st.expander('Bounded transient PAZy context and cross-source verification needs'):
            render_multi_study(view)
    st.markdown('#### Remaining uncertainties')
    primary, rest = prioritize_limitations(view)
    for limit in primary: st.write('• '+limit)
    with st.expander('All remaining warnings, evidence gaps and execution details'):
        for limit in rest: st.write('• '+limit)
        for gap in view['gaps']: st.write(f"{gap['code']}: {gap['requirement']}")
        for issue in view['issues']: st.warning(issue)
        st.text('Scientific partial reasons: '+str(view['advice']['partial_reasons']))
        limits = view.get('limits')
        model_attempts = str(view['model_requests'])
        tool_attempts = str(view['tool_attempts'])
        if limits:
            model_attempts += f" / {limits.get('turn_models', DEFAULT_TURN_MODELS)}"
            tool_attempts += f" / {limits.get('turn_tools', DEFAULT_TURN_TOOLS)}"
        st.text(f"Workflow {view['workflow_status']} · backend attempts {model_attempts} · scientific attempts {tool_attempts}")
    st.markdown('#### Supporting publications')
    st.caption('Original studies represented in the selected evidence; a shared DOI is not independent replication or authenticated source-cell equivalence.')
    dois = list(dict.fromkeys(c['publication_doi'] for c in view.get('multi_study',{}).get('pazy',{}).get('records',[])))
    for doi in dois:
        url = doi_link(doi)
        if url: st.link_button('Original experiment · '+doi, url)
    if not dois: st.write('No original-study DOI was returned in the bounded context; publication-to-cell mapping remains unverified.')
    st.link_button('NREL source dataset', 'https://zenodo.org/records/15417757')
    st.link_button('PAZy source resource', 'https://www.pazy.eu/')
    st.link_button('PAZy database article · resource attribution', 'https://doi.org/10.1093/nar/gkag1026')
    st.caption('Additional suggested reading: none retrieved. No new literature or PAZy API request was made. This result disappears on rerun.')
