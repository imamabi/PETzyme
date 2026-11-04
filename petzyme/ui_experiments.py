"""Transient combined experimental view over existing allowlisted public projections."""

from __future__ import annotations

import re
from dataclasses import dataclass

import streamlit as st

from .contracts import UnitRepresentation
from .evidence_records import CombinedExperimentalPreview, experimental_records
from .experimental_coverage import ExperimentalCoverage
from .ui_findings import display_response

# Ten rows allow five source-labelled subjects to show both sources; omissions stay explicit.
VISIBLE_EXPERIMENT_ROWS = 10


@dataclass(frozen=True)
class ExperimentalRow:
    enzyme: str
    source: str
    form: str
    temperature: str
    ph: str
    duration: str
    response: str
    unit: str
    original_response: str
    reference: str
    definition: str
    original_unit: str = ""
    conversion: UnitRepresentation | None = None
    row_linkage: str = ""
    reported_replicate_n: int | None = None


def _value(value: object) -> str:
    return "Not recorded" if value is None else str(value)


def combined_experimental_rows(value: CombinedExperimentalPreview) -> list[ExperimentalRow]:
    """Flow: Receives allowlisted multi-study preview from accumulate_pet_results(); returns bounded ExperimentalRow list for render_combined_experiments(), preserving each measurement and source reference without inference, pooling or storage."""
    rows = []
    for record in experimental_records(value):
        conversion = record['qualified_unit']
        converted = bool(conversion and conversion['status']=='INFERRED_SUPPORTED')
        display = str(conversion['standardized_value']) if converted and conversion else record['original_response']
        duration = (str(record['duration_value'])+' '+str(record['duration_unit'] or 'unit not recorded')
                    if record['duration_value'] is not None else 'Not recorded')
        rows.append(ExperimentalRow(record['enzyme'],
            'NREL · inferred conversion' if converted else 'NREL · original CSV' if record['source']=='NREL' else 'PAZy · reported',
            record['pet_form'], _value(record['temperature_c']), _value(record['ph']), duration,
            display_response(display), conversion['display_unit'] if converted and conversion else record['original_unit'],
            record['original_response'], record['reference'], record['measurement_definition'],
            record['original_unit'], conversion, record['row_linkage'],
            record['replicate_information'].get('replicate_n') if type(record['replicate_information'].get('replicate_n')) is int else None))
    return rows


def _measurement_index(row: ExperimentalRow) -> str:
    """Flow: Receives ExperimentalRow from combined_experimental_rows(); returns a transient source-position label for render_combined_experiments()."""
    match = re.search(r"#/(\d+)/assays/(\d+)/measurements/(\d+)$", row.reference)
    if match:
        return f"P{match[1]} A{match[2]} M{match[3]}"
    nrel = re.search(r"NREL CSV record (\d+), activity_at_(.+)$", row.reference)
    if nrel:
        return f"row {nrel[1]} / {nrel[2]}"
    return "Source reference in Details"


def render_combined_experiments(value: CombinedExperimentalPreview) -> None:
    """Flow: Receives transient multi-study preview from render_result(); returns None after displaying source-specific ExperimentalRow values and executed coverage counts, without cache, exports, session writes or new inference."""
    st.markdown("#### What has been tested?")
    inventories: list[ExperimentalCoverage | None] = list(value.get("experimental_coverages") or [])
    if not inventories:
        inventories.append(value.get("experimental_coverage"))
    for inventory in inventories:
        if not inventory:
            continue
        st.write(
            f"**{inventory['enzyme']}** · NREL: {inventory['nrel_observations']} observations; "
            f"PAZy: {inventory['pazy_assays']} assays, {inventory['pazy_measurements']} measurements "
            f"from {inventory['pazy_publications']} publication(s)."
        )
        st.caption(
            'NREL excluded by explicit source scope; cross-source additions are not assessed.' if inventory.get('source_scope') == 'PAZY_ONLY' else
            f"Nominal overlap: {inventory['nominal_overlap']} conditions. "
            f"PAZy adds {inventory['pazy_additional_conditions']} condition combinations; "
            f"NREL-only: {inventory['nrel_only_conditions']}. Identity and assay equivalence are unverified."
        )
        if inventory['same_publication_assays']:
            st.write(
                f"**Why it matters:** {inventory['same_publication_assays']} of "
                f"{inventory['pazy_assays']} PAZy assays in this coverage cite the NREL study. "
                "Their appearance in both sources does not establish independent replication; "
                "individual experiment and construct linkage remain unverified."
            )
        if inventory["nrel_candidate_records"] > 1 or inventory["pazy_candidate_records"] > 1:
            st.warning(
                "This label matches multiple candidate source records. These coverage counts do not resolve them into one protein or construct."
            )

    st.markdown("#### What were the results? · MEASURED / COMPUTED")
    rows = combined_experimental_rows(value)
    product_rows = [r for r in rows if r.unit == "µmol aromatic products/mg enzyme"]
    preferred = (
        [r for r in rows if r.source.startswith("NREL") or r in product_rows]
        if product_rows
        else rows
    )
    # Give each returned source-labelled enzyme a visible row before showing
    # additional measurements; exact measurements remain in Details.
    first, rest, seen = [], [], set()
    for row in preferred:
        key = (row.enzyme, row.source)
        (rest if key in seen else first).append(row)
        seen.add(key)
    shown = (first + rest)[:VISIBLE_EXPERIMENT_ROWS]
    if shown:
        st.table(
            [
                {
                    "Enzyme": r.enzyme,
                    "Source": r.source,
                    "Measurement index": _measurement_index(r),
                    "Metric": r.definition.split(' · ', 1)[0] if r.source.startswith('PAZy') else 'CSV response',
                    "PET form": r.form,
                    "°C": r.temperature,
                    "pH": r.ph,
                    "Duration": r.duration,
                    "Response": r.response,
                    "Unit": r.unit,
                }
                for r in shown
            ]
        )
    else:
        st.write(
            "No measured response was retrieved under these filters. Missing measurements are not zero."
        )
    nrel_preview = len(value['nrel']['observations'])
    assays_preview = len(value['pazy']['records'])
    pazy_preview = sum(len(assay['measurements']) for assay in value['pazy']['records'])
    total_nrel = value['nrel'].get('total_matches')
    total_assays = value['pazy'].get('total_matches')
    total_measurements = value['pazy'].get('total_matching_measurements')
    if type(total_nrel) is int and type(total_assays) is int and type(total_measurements) is int:
        st.caption(f"Current source filters match {total_nrel} NREL observation{'s' if total_nrel != 1 else ''} and {total_assays} PAZy assay{'s' if total_assays != 1 else ''} containing {total_measurements} individual measurement{'s' if total_measurements != 1 else ''}.")
    else:
        st.caption("For multiple source queries, matching totals are reported per query in Details and are not summed.")
    st.caption(
        f"Retrieved preview: {nrel_preview} NREL observation{'s' if nrel_preview != 1 else ''} and {assays_preview} PAZy assay{'s' if assays_preview != 1 else ''} "
        f"containing {pazy_preview} individual measurement{'s' if pazy_preview != 1 else ''}; {len(shown)} of {len(rows)} "
        "retrieved measurement rows shown. These are bounded previews, not all matching records. "
        "Original values and source references are in Details; no responses are pooled."
    )
    st.caption(
        "NREL inferred conversion: CSV mmol/mg is INFERRED from worksheet reconciliation; displayed µmol/mg = original ×1000 (COMPUTED). Original CSV declaration of units remains unverified. PAZy reported values use source-reported units without conversion."
    )
    st.caption("PAZy P/A/M labels are zero-based archive positions, not native assay IDs or proof of replicate relationships.")
    if any(r.source.startswith('NREL') for r in rows) and any(r.source.startswith('PAZy') for r in rows):
        st.caption("Matching conditions or worksheet references do not prove record-level equivalence. Differences between source values remain unresolved unless independently verified.")
    st.caption(
        "PAZy is publication-derived curation. Shared publications are not independent replication. "
        "NREL duration and units are not filled from PAZy. Full preview counts and omissions are in Details."
    )

    st.markdown("#### What is interesting or missing? · COMPUTED")
    for inventory in inventories:
        if not inventory:
            continue
        if inventory['other_publication_assays']:
            st.write(
                f"• {inventory['other_publication_assays']} PAZy assays cite other studies. "
                "Different publications alone do not prove independent experiments."
            )
        if inventory["pazy_additional_conditions"] and inventory.get("source_scope") != "PAZY_ONLY":
            additions = [
                c
                for c in inventory["conditions"]
                if c["pazy_assays"] and not c["nrel_observations"]
            ]
            for condition in additions[:3]:
                st.write(
                    f"• PAZy adds: {condition['temperature_c']}°C, pH {condition['ph']}, "
                    f"{condition['morphology']} {condition['form']}; this is documented, not a novel condition."
                )
            st.caption(
                f"{max(0, inventory['pazy_additional_conditions'] - min(3, len(additions)))} additional combinations not listed here."
            )
        if inventory['domain_combinations']:
            st.caption(
                f"Observed-condition grid: {inventory['domain_temperature_count']} temperatures × "
                f"{inventory['domain_ph_count']} pH values × {inventory['domain_material_count']} "
                f"PET form/morphology pairs = {inventory['domain_combinations']} combinations; "
                f"{inventory['documented_combinations']} have selected-source records. "
                f"{inventory['records_with_unknown_condition']} observations/assays have incomplete condition metadata."
            )
        if inventory["missing_in_both_count"]:
            selected_sources = (
                "the selected PAZy scope"
                if inventory.get("source_scope") == "PAZY_ONLY"
                else "both selected sources"
            )
            st.write(
                f"• {inventory['missing_in_both_count']} combinations lack records in {selected_sources} "
                "within the observed temperature/pH/material domain; absence here does not establish novelty or feasibility."
            )
        elif inventory['domain_combinations']:
            st.write(
                "• Every combination in this limited observed-condition grid has a source record. "
                "This does not establish complete coverage of other temperatures, pH values, materials, durations or protocols."
            )
        else:
            st.write(
                "• No source observations define a coverage domain for this label; identity or evidence remains unresolved."
            )
        with st.expander("Condition coverage and scope · " + inventory["enzyme"]):
            st.caption(inventory["basis"])
            for condition in inventory["conditions"]:
                st.text(
                    f"{condition['temperature_c']}°C · pH {condition['ph']} · {condition['morphology']} {condition['form']} "
                    f"· NREL {condition['nrel_observations']} · PAZy {condition['pazy_assays']} assays / {condition['pazy_measurements']} measurements"
                )
            for condition in inventory["missing_in_both"]:
                st.text(
                    f"No selected-source record: {condition['temperature_c']}°C · pH {condition['ph']} · {condition['morphology']} {condition['form']}"
                )
            st.caption(
                f"Coverage rows omitted: {inventory['conditions_omitted']}; missing combinations omitted: {inventory['missing_omitted']}."
            )
    with st.expander("Exact responses, definitions and assay references"):
        for row in rows:
            st.text(
                f"{row.source} · {row.enzyme} · {row.temperature}°C · pH {row.ph} · {row.form} "
                f"· {row.duration} · ORIGINAL {row.original_response} {row.original_unit or row.unit}"
            )
            st.text(row.definition + " · " + row.reference)
            if row.reported_replicate_n is not None:
                st.caption(f"Source reports replicate_n={row.reported_replicate_n}; that count alone does not identify which measurements are replicates.")
            if row.conversion:
                st.text("Working representation: " + str(row.conversion))
            if row.row_linkage:
                label = "Documentary linkage assessment: " if row.source.startswith("NREL") else "Source-reported evidence location: "
                st.text(label + row.row_linkage)
        st.caption(
            "PAZy source: https://www.pazy.eu/ · database article: https://doi.org/10.1093/nar/gkag1026. "
            "Transient display only; no exports or result history. Same-study curation is not independent replication."
        )
    with st.expander("Unit reconciliation · inferred conversions and unresolved records"):
        for label, explanation in value.get("units", {}).items():
            if label == "lcc_rounding_diagnostics" and isinstance(explanation, list):
                st.markdown("**Four LCC-ICCG alternate-rounding discrepancies**")
                st.caption(
                    "These match the adopted U/volume reconstruction; they are not failed unit conversions. The alternate rounded-V method disagrees. Construct and original aggregation authentication remain separate."
                )
                for case in explanation:
                    if isinstance(case, dict):
                        st.text(str(case))
            elif label == "record_examples" and isinstance(explanation, list):
                for example in explanation:
                    st.text(str(example))
            else:
                st.write(label.replace("_", " ").capitalize() + ": " + str(explanation))
