"""Transient multi-study research display, with no table exports or record-bearing widgets."""
from __future__ import annotations

import streamlit as st
from .runtime_config import DEFAULT_TURN_MODELS, DEFAULT_TURN_TOOLS
from .ui_adapter import ResearchView
from .ui_findings import display_response
from .ui_pazy import doi_link


def render_multi_study(view: ResearchView) -> None:
    """Flow: Receives transient ResearchView from present_investigation() through render_findings(); renders final bounded multi-study evidence without session history, file links or exports."""
    value = view['multi_study']
    st.divider()
    st.subheader('Multi-Study Evidence')
    st.caption('Deterministic tool conclusions · COMPUTED. This summary is not model-authored reasoning.')
    st.info(value['answer'])
    limits = view.get('limits')
    model_attempts = str(view['model_requests'])
    tool_attempts = str(view['tool_attempts'])
    if limits:
        model_attempts += f" / {limits.get('turn_models', DEFAULT_TURN_MODELS)}"
        tool_attempts += f" / {limits.get('turn_tools', DEFAULT_TURN_TOOLS)}"
    st.caption(f"Workflow {view['workflow_status']} · backend attempts {model_attempts} · scientific attempts {tool_attempts}")
    application_cohort = any(update.get('actor') == 'APPLICATION_COVERAGE' and update['call_id'].startswith('cohort_') for update in view['progress'])
    st.caption('Deterministic offline tool selection; real local evidence; live Muse requests: 0.' if view['mode']=='DETERMINISTIC_OFFLINE' else
               ('Application-replayed cohort tools; Muse received bounded evidence. See provider usage for actual HTTP activity.' if application_cohort else
                'Muse tool selection; backend attempts are not independently observed HTTP counts or token usage.') if view['mode']=='LIVE_META' else
               'Mocked/in-process tool selection; real local evidence; live Muse requests: 0. This is not live Muse acceptance.')
    st.warning('Bounded evidence selection. Execution completion does not establish activity superiority. PAZy previews disappear on rerun; no history or export.')
    st.markdown('#### Source-specific experimental evidence')
    nrel, pazy = value['nrel'], value['pazy']
    aggregate = value.get('accumulation')
    multiple = aggregate and aggregate['run_count'] > 1
    if multiple:
        st.text(f"Unique NREL cells retrieved: {aggregate['unique_nrel_retrieved']}; displayed: {nrel['returned']}; display omissions: {aggregate['display_omitted_nrel']}.")
    else:
        st.text(f"NREL original observations: {nrel['returned']} / {nrel['total_matches']}; truncated: {nrel['truncated']}.")
    for o in nrel['observations']:
        st.text(f"MEASURED · NREL {o['enzyme_identifier']} · {o['temperature_c']}°C · pH {o['ph']} · {o['pet_form']} · recorded response {display_response(o['original_text'])} · units {o['unit_status']}")
    with st.expander('Exact original NREL source strings'):
        for o in nrel['observations']:
            st.text(f"Record {o['source_row']}, {o['source_column']}: {o['original_text']} · CSV SHA-256 {o['csv_sha256']}")
    if multiple:
        st.text(f"Unique PAZy assay projections retrieved: {aggregate['unique_pazy_assays_retrieved']}; displayed: {pazy['returned']}; unique measurements retrieved: {aggregate['unique_pazy_measurements_retrieved']}.")
        st.text('Display omissions · assays: '+str(aggregate['display_omitted_assays'])+'; measurements: '+str(aggregate['display_omitted_measurements']))
        st.caption(aggregate['interpretation'])
    else:
        st.text(f"PAZy publication-derived curation: {pazy['returned']} / {pazy['total_matches']} assays; {pazy['matching_parent_records']} parents; {pazy['total_matching_measurements']} matching measurements; {pazy['matching_publications']} publications. Truncated: {pazy['truncated']}.")
    st.caption(pazy['selection'])
    if not pazy['records']: st.write('No exact PAZy match under these filters. This is not proof that the enzyme or experimental evidence is absent.')
    for card in pazy['records']:
        st.text(f"PAZy parent {card['parent_index']}, protein {card['protein_id']} ({card['protein_name']}), assay {card['assay_index']} · {card['locator']}")
        for m in card['measurements']:
            r = m['record']
            st.text(f"MEASURED / PUBLICATION-DERIVED · measurement {m['measurement_index']}: {r.get('qualifier')} {r.get('value')} {r.get('unit')} · metric {r.get('metric_type')} · analyte {r.get('analyte')} · categorical {r.get('categorical_result')} · reported {r.get('reported_result')}")
            st.text(f"Reported replicates: {r.get('replicate_n')} · uncertainty {r.get('uncertainty_value')} ({r.get('uncertainty_type')}) · digitized {r.get('value_is_digitized')}")
        if multiple:
            st.text(f"Distinct measurements displayed for this source assay: {len(card['measurements'])}; source total: {card['total_measurements']}. Per-call omissions and out-of-scope counts remain below.")
        else:
            st.text(f"Measurements omitted from this assay: {card['omitted_measurements']}; outside the requested measurement selection: {card['out_of_scope_measurements']}.")
        st.text('Missing assay fields: '+', '.join(card['missing_fields']))
    if aggregate:
        coverage=aggregate['condition_coverage']
        st.markdown('#### Missing evidence and condition coverage')
        nrel_excluded = any(receipt['source'] == 'NREL' and receipt['status'] == 'EXCLUDED'
                            for receipt in view.get('source_coverage', []))
        if nrel_excluded:
            st.text('NREL condition coverage was not assessed because NREL was excluded.')
        else:
            st.text(f"Actual NREL requested-condition targets retrieved: {coverage['covered']} / {coverage['requested']}; complete: {coverage['complete']}.")
            st.caption(coverage['basis'])
            if coverage['unspecified_fields']:
                st.text('Unspecified constraints: '+str(coverage['unspecified_fields'])+'; this is not exhaustive experimental coverage.')
            for target in coverage['missing_conditions']:
                st.text('Missing requested target: '+str(target))
            if coverage['unlisted_conditions']:
                st.text('Additional targets not listed: '+str(coverage['unlisted_conditions']))
        if aggregate['source_conflicts']: st.error('Conflicting source representations: '+str(aggregate['source_conflicts'])+'; no reconciliation or pooling performed.')
        with st.expander('Coverage by executed scientific call'):
            for i,call in enumerate(aggregate['calls'],1):
                st.text(f"Call {i} · NREL {call['nrel']} · PAZy {call['pazy']} · partial reasons {call['partial_reasons']}")
                st.text('Actually retrieved NREL targets: '+str(call['conditions']['covered_conditions']))
                st.text('Call-bound identity assessment: '+str(call['identity']))
                for assay in call['assays']: st.text('Assay-scoped omissions: '+str(assay))
    st.markdown('#### Conditions, units and measurement definitions')
    for card in pazy['records']:
        c,m=card['conditions'],card['material']
        st.text(f"{card['locator']} · temperature {c['temperature_c']}°C · pH {c['ph']} · duration {c['duration_value']} {c['duration_unit']}")
        st.text(f"PET form {m['form']} · morphology {m['morphology']} · crystallinity {m['crystallinity_percent']}% · pretreatment {m['pretreatment']}")
        with st.expander('Additional documented protocol · '+card['locator']):
            for key,item in {**c,**m}.items(): st.text(key.replace('_',' ')+': '+str(item))
            for method in card['detection_methods']: st.text('Detection method: '+str(method))
    for label, description in value['units'].items(): st.text(label+': '+str(description))
    if nrel['unverified_filter_fields']: st.text('NREL cannot authenticate these filters: '+str(nrel['unverified_filter_fields']))
    st.markdown('#### Publication references and source locators')
    st.link_button('NREL dataset · original source', 'https://zenodo.org/records/15417757')
    st.link_button('PAZy · source resource', 'https://www.pazy.eu/')
    st.link_button('PAZy database article (2026)', 'https://doi.org/10.1093/nar/gkag1026')
    st.caption('Pinned activity snapshot · schema '+pazy['schema_version']+' · SHA-256 '+pazy['archive_sha256'])
    for card in pazy['records']:
        st.text(card['locator']+' · source evidence: '+str(card['evidence_locator']))
        url=doi_link(card['publication_doi'])
        if url: st.link_button('Original study · '+card['publication_doi'],url)
    st.markdown('#### Verified and unresolved overlaps')
    st.text('Current parent identity assessment: '+str(value['identity']))
    st.text(value['overlap_interpretation'])
    st.text('Computed archive-wide coverage (not independent experiments): '+str(value['coverage']))
    st.markdown('#### Comparability conclusions')
    if multiple:
        st.text(f"Unique executed comparison assessments retrieved: {aggregate['unique_comparisons_retrieved']}; displayed: {len(value['comparisons'])}; display omissions: {aggregate['display_omitted_comparisons']}.")
        st.caption('Only actually executed source-pair assessments are retained. No additional cross-call comparisons were synthesized.')
    else:
        st.text('Classification coverage within returned previews: '+str(value['comparison_coverage']))
    for c in value['comparisons']:
        st.text(f"{c['pazy_locator']} vs NREL {c['nrel_record']} / {c['nrel_column']}: {c['category']} · incompatible fields {c['incompatible_fields']} · missing {c['missing_fields']}")
    if not value['comparisons']: st.text('No bound cross-source candidate pair in the returned selection; no quantitative comparison is established.')
    st.markdown('#### Critical evidence gaps and verification recommendations')
    for gap in value['gaps']: st.text(f"{gap['priority']}. {gap['code']}: {gap['requirement']}")
    for text in value['missing_work']: st.text(text)
    for text in dict.fromkeys(view['limitations']+view['issues']): st.text(text)
    st.caption(value['context_projection'])
    if value['source_preview_omissions']: st.text('Additional preview omissions: '+str(value['source_preview_omissions']))
