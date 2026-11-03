"""Concise research-result hierarchy over unchanged, validated public findings."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import streamlit as st
import streamlit.components.v1 as components

from .enzyme_discovery import EnzymeDiscovery
from .ui_adapter import ResearchView
from .ui_experiments import CombinedExperimentalPreview, render_combined_experiments
from .ui_findings import display_response, evidence_snapshot, result_overview, similarity_snapshot


def render_discovery(view: ResearchView) -> None:
    """Flow: Receives transient ResearchView from render_result(); returns None after rendering the executed EnzymeDiscovery answer first, exact measurements and provenance in collapsed details, without persistence or inference."""
    value: EnzymeDiscovery = view['discovery']
    local = view['mode'] == 'DETERMINISTIC_LOCAL_DISCOVERY'
    st.subheader('Research findings', anchor='research-findings')
    st.caption('Question answered: ' + view.get('submitted_question', view['question']))
    st.caption('Run ' + view.get('run_id', 'unavailable') +
               (' · deterministic local scientific tool · no Muse inference' if local else
                ' · Muse tool orchestration; ranking computed by the deterministic local scientific tool'
                if view['mode'] == 'LIVE_META' else
                ' · Mock/offline orchestration; not live Muse acceptance. Ranking computed by the local scientific tool.'))
    st.info(value['answer'])
    if view.get("scientific_scope", {}).get("background_only"):
        st.caption("Reported PET endpoint screen only: the desired performance metric is unspecified. Clarify that metric before treating this as a direct quantitative comparison; no global activity ranking or optimum is established.")
    group = value['group']
    if not group:
        st.warning('No eligible comparable measurements matched the filters. Try another documented condition group.')
        return
    st.caption('CONDITIONAL WITHIN-GROUP RANKING · PARTIAL · COMPUTED means; not a global activity ranking')
    st.table([{'Enzyme': e['enzyme'], 'Mean response': display_response(str(e['response'])),
               'Unit': e['unit'], '°C': f"{e['temperature_c']:g}", 'pH': f"{e['ph']:g}",
               'PET form': e['pet_form'].replace('_', ' ').lower(), 'Source': 'PAZy curation'}
              for e in value['entries']])
    st.caption(f"{len(value['entries'])} of {group['eligible_enzymes']} eligible source-labelled enzymes shown; "
               f"{value['omitted_enzymes']} omitted. Endpoint: {group['duration_value']} {group['duration_unit']}. "
               'Each mean uses two documented biological duplicates within one assay. No cross-assay pooling or conversion.')
    st.caption('Group chosen by documented coverage, not by highest response. Specify a temperature, pH or PET form to inspect another group.')
    st.write(('Returned NREL original values are available as nominal source references in Details. '
              if any(e['nrel_references'] for e in value['entries']) else
              'No authenticated NREL reference was returned for these displayed entries; valid PAZy evidence remains available. ')
             + 'NREL units are unverified and do not contribute to the ranking. '
             'Shared publication curation is not independent replication; enzyme identities are source labels.')
    if view.get("multi_study"):
        render_combined_experiments(cast(CombinedExperimentalPreview, view["multi_study"]))
    if view["plan"]:
        st.markdown("#### Suggested next step")
        for index, step in enumerate(result_overview(view).steps, 1):
            st.markdown(f"**{index}. {step.text}**")
            st.caption(step.label + " · " + step.rationale)
    if view.get('synthesis', {}).get('status') == 'INCOMPLETE':
        st.warning(('Muse' if view['mode'] == 'LIVE_META' else 'The offline/mock model')
                   + ' did not complete a validated explanation. The deterministic database ranking above remains available.')
    with st.expander('Exact measurements, assay group and source provenance', expanded=False):
        st.write(group['selection_rule'])
        st.write(group['measurement_definition'])
        for key, value_ in group['protocol'].items():
            st.text(key.replace('_', ' ') + ': ' + str(value_))
        st.link_button('Original experimental publication', 'https://doi.org/' + group['publication_doi'])
        for entry in value['entries']:
            st.text(entry['enzyme'] + ' · COMPUTED mean: ' + repr(entry['response']) + ' ' + entry['unit'])
            for measurement in entry['measurements']:
                st.text(f"MEASURED biological duplicate {measurement['biological_duplicate']}: "
                        f"{measurement['value']!r} {measurement['unit']} · {measurement['locator']}")
            st.text(entry['workbook_reference'])
            for reference in entry['nrel_references']:
                st.text(f"NREL name/condition candidate only: {reference['original_response']} · "
                        f"{reference['unit_status']} · record {reference['source_row']} · {reference['source_column']}")
        for limitation in view['limitations']:
            st.write('• ' + limitation)
        st.caption(f"{value['eligible_groups']} eligible assay groups; {value['excluded_assays']} examined assays excluded by eligibility or ambiguity checks.")
        st.link_button('PAZy resource', 'https://www.pazy.eu/')
        st.link_button('PAZy database article', 'https://doi.org/10.1093/nar/gkag1026')
        st.link_button('NREL dataset', 'https://zenodo.org/records/15417757')
        st.caption(f"Transient display only. No exports or result history. Backend model attempts: {view['model_requests']}; scientific attempts: {view['tool_attempts']}.")


def focus_result() -> None:
    """Flow: Receives the consumed submission-only focus flag from render_findings(); returns None after bounded navigation to the result heading, without inference, scientific state changes or external requests."""
    # Fixed UI code only: never interpolate researcher, source or provider text into JavaScript.
    components.html(
        """<script>
    (() => { let remaining = 20;
      function focus() { try {
        const target = window.parent.document.getElementById('research-findings');
        if (target) { target.style.scrollMarginTop = '4rem'; target.scrollIntoView({block:'start'}); return; }
        if (--remaining > 0) requestAnimationFrame(focus);
      } catch (_) { /* The explicit findings link remains available if a browser blocks parent access. */ } }
      focus();
    })();
    </script>""",
        height=0,
        tab_index=-1,
    )


def render_result(view: ResearchView, render_details: Callable[[], None]) -> None:
    """Flow: Receives ResearchView and complete-detail callback from render_findings(); returns None after displaying result_overview()/evidence_snapshot() followed by collapsed original scientific details, without new inference or persistence."""
    scope = view.get("scientific_scope")
    qualified_pet_screen = bool(scope and not scope["issues"]
        and scope["request"]["material"] == "PET"
        and scope["request"]["quantity"] in ("UNSPECIFIED", "RECORDED_RESPONSE"))
    if view.get('discovery') and (not (scope and scope['background_only']) or qualified_pet_screen):
        render_discovery(view)
        with st.expander('Details · complete evidence, sources and execution', expanded=False):
            render_details()
        return
    overview = result_overview(view)
    st.subheader("Research findings", anchor="research-findings")
    st.caption('Question answered: ' + view.get('submitted_question', view['question']))
    st.caption('Run ' + view.get('run_id', 'unavailable'))
    with st.container(border=True):
        st.caption(overview.label)
        st.write(view['headline'] if view.get('comparison_groups') else overview.answer)
        st.caption(overview.attribution)
    if view.get('comparison_groups'):
        st.caption('Historical groups are shown separately. No cross-group ranking, pooling, causal effect or independent replication is inferred.')
        columns = st.columns(len(view['comparison_groups']))
        for column, group in zip(columns, view['comparison_groups']):
            with column:
                st.markdown('#### Historical group: ' + str(group['question']))
                render_combined_experiments(cast(CombinedExperimentalPreview, group['preview']))
        if view.get('group_assessments'):
            st.markdown('#### NREL record-level comparability · COMPUTED')
            summaries: dict[tuple[str, str | None, tuple[str, ...], tuple[str, ...]], list[str]] = {}
            for assessment in view['group_assessments']:
                key = (assessment['category'], assessment['varying_variable'],
                       tuple(assessment['missing_fields']), tuple(assessment['blocking_differences']))
                summaries.setdefault(key, []).append(assessment['label'])
                numeric = assessment.get('numerical_comparison')
                if assessment['direct_comparison_supported'] and numeric is not None:
                    st.write('**' + assessment['label'] + '**')
                    st.table([{'Recorded A': numeric.get('original_a'), 'Recorded B': numeric.get('original_b'),
                               'Difference B − A': numeric.get('difference_b_minus_a'), 'Ratio B / A': numeric.get('ratio_b_over_a'),
                               'Change from A (%)': numeric.get('percent_change_from_a'), 'Unit': numeric.get('unit')}])
                    st.caption('COMPUTED descriptive values based on source-bound caller-supplied metadata; no statistical significance or causal effect is established.')
            for (category, varying, missing_fields, differences), labels in summaries.items():
                st.write('**' + ', '.join(labels) + '**: ' + str(category).replace('_', ' ').lower() + '.')
                missing = ', '.join(missing_fields) or 'none reported'
                different = ', '.join(differences) or 'none reported'
                st.caption(f"Varied variable: {varying or 'none specified'}. Missing: {missing}. Blocking differences: {different}.")
            st.caption('NREL source cells are assessed independently; PAZy identity is unverified.')
    if view.get('multi_study'):
        render_combined_experiments(cast(CombinedExperimentalPreview, view['multi_study']))
        if view.get('cohort_missing'):
            st.info("No bounded matching records were returned for: " + ", ".join(view['cohort_missing']) + ". This does not establish absence from all studies.")
        if view.get('cohort_errors'):
            st.warning("Evidence retrieval failed for: " + ", ".join(view['cohort_errors']) + ". No absence was inferred.")
    if view.get('restricted_annotation_status'):
        st.info("PAZy biological-annotation tools were run. The current privacy policy does not allow their protein or publication details to be displayed here; tool status is available in Details. This does not mean that PAZy returned no annotations.")
        st.link_button("Open the official PAZy resource", "https://www.pazy.eu/")
    if overview.blocking:
        st.error("The requested comparison is unsupported. PET evidence below is background only.")

    st.markdown("#### Suggested next step")
    if overview.steps:
        labels = {step.label for step in overview.steps}
        common_label = overview.steps[0].label if len(labels) == 1 else None
        if common_label:
            st.caption(common_label)
        first = overview.steps[0]
        prefix = "Conditional recommendation: " if first.is_experiment else ""
        st.markdown(f"**{prefix}{first.text}**")
        st.caption((first.label + " · " if common_label is None else "") + first.rationale)
        if len(overview.steps) > 1:
            with st.expander("Other actions and qualifications"):
                for step in overview.steps[1:]:
                    prefix = "Conditional recommendation: " if step.is_experiment else ""
                    st.markdown(f"**{prefix}{step.text}**")
                    st.caption((step.label + " · " if common_label is None else "") + step.rationale)
    if overview.starting_points:
        st.markdown("**Observed starting points for discussion · MEASURED**")
        for point in overview.starting_points:
            st.write("• " + point)
        st.caption(
            "Examples from returned observations, not ranked recommendations. Laboratory suitability is unverified. "
            "A highest recorded response is not an established optimum; any future experiment remains hypothetical."
        )

    st.markdown("#### What remains uncertain")
    for uncertainty in overview.uncertainties:
        st.write("• " + uncertainty)
    if not overview.uncertainties:
        st.caption("See the complete scientific limitations and missing evidence in Details.")

    groups = [] if view.get('multi_study') else evidence_snapshot(view)
    if not view.get('multi_study'):
        st.markdown("#### Evidence snapshot · " + ("COMPUTED" if view["similarity"] else "MEASURED"))
    for group in groups:
        st.caption(group.name + " · units " + group.units)
        # Static tables have no download controls and contain only allowlisted NREL display fields.
        st.table(
            [
                {
                    "Enzyme": row["enzyme"],
                    "°C": row["temperature"],
                    "pH": row["ph"],
                    "Response": row["response"],
                }
                for row in group.rows
            ]
        )
        st.caption(
            f"{len(group.rows)} of {group.total} returned observations shown for this form. "
            "Returned order; no new activity ranking; exact originals and remaining rows are in Details."
        )
    if view["similarity"]:
        neighbors = similarity_snapshot(view)
        st.table([{"Protein": n["protein"], "Cosine similarity": n["cosine"]} for n in neighbors])
        st.caption(
            "Computed representation similarity only; not PET activity, sequence identity or a whole-dataset search. Full bounded-search coverage is in Details."
        )
    elif not groups and not view.get('multi_study') and not view.get('comparison_groups') and not view.get('restricted_annotation_status'):
        st.caption("No measured NREL response was returned. Missing measurements are not zero.")
    if view["comparison"]:
        st.caption(
            "COMPUTED assessment · "
            + str(view["comparison"]["category"]).replace("_", " ").capitalize()
        )
        numeric = view['comparison'].get('numerical_comparison')
        if view['comparison'].get('direct_comparison_supported') is True and isinstance(numeric, dict):
            st.table([{'Recorded A': numeric.get('original_a'), 'Recorded B': numeric.get('original_b'),
                       'Difference B − A': numeric.get('difference_b_minus_a'), 'Ratio B / A': numeric.get('ratio_b_over_a'),
                       'Change from A (%)': numeric.get('percent_change_from_a'), 'Unit': numeric.get('unit')}])
            st.caption('COMPUTED descriptive values from source-bound caller-supplied metadata; no statistical significance or causal effect is established.')
    if groups:
        st.caption(
            "Rounded display only; any inferred conversion is explicitly labelled COMPUTED. Forms are kept separate. Exact originals and provenance are in Details."
        )

    with st.expander("Details · complete evidence, sources and execution", expanded=False):
        render_details()
