"""Allowlisted, bounded transient presentation shared by model facts and UI updates."""
from __future__ import annotations

from typing import Callable, TypedDict, cast

from .contracts import JSONValue


class SummaryMeasurement(TypedDict):
    record: dict[str, JSONValue]


class AssaySummaryInput(TypedDict):
    conditions: dict[str, JSONValue]
    material: dict[str, JSONValue]
    measurements: list[SummaryMeasurement]
    protein_name: str
    publication_doi: str
    omitted_measurements: int | None


def measurement_summary(card: AssaySummaryInput) -> str:
    """Flow: Receives an allowlisted AssaySummaryInput from preview_pet_evidence() or accumulate_pet_results(); returns exact source-bound summary str for their canonical communication/display, never averaging individual measurements or inventing omission counts."""
    conditions, material = card["conditions"], card["material"]
    measurements = "; ".join(
        str(m["record"].get("qualifier") or "")
        + " "
        + str(
            m["record"].get("value")
            if m["record"].get("value") is not None
            else m["record"].get("reported_result")
            or m["record"].get("categorical_result")
            or "not reported"
        )
        + " "
        + str(m["record"].get("unit") or "unit unreported")
        + " ["
        + str(m["record"].get("metric_type") or "definition unreported")
        + "]"
        for m in card["measurements"]
    )
    omitted = (
        str(card["omitted_measurements"])
        if card["omitted_measurements"] is not None
        else "unknown after selection merge"
    )
    return (
        f"PAZy publication-derived measurements for {card['protein_name']}: "
        f"{conditions.get('temperature_c')}°C, pH {conditions.get('ph')}, {material.get('morphology')} {material.get('form')}, "
        f"{conditions.get('duration_value')} {conditions.get('duration_unit')}: {measurements}. "
        f"Source: {card['publication_doi']}; omitted measurements: {omitted}. "
        "Individual reported endpoints; not independent replication or verified cross-source construct identity."
    )


def preview_pet_evidence(value: dict[str, JSONValue], text: Callable[[object], str]) -> dict[str, JSONValue]:
    """Project a multi-study result into a bounded allowlisted preview.

    Purpose: shared presenter for model facts and UI updates. Keeps
        aggregate counts, coverage, identity status, comparisons (at
        most 8), gaps, units, and support flags whole; bounds source
        strings (300 chars) and arrays (3 items) with every shortening
        recorded in 'source_preview_omissions'. Warnings, gaps, and
        missing fields are never truncated.
    Flow: Receives result dict from investigate_pet_evidence() and safe_text() Callable; returns bounded dict[str, JSONValue] for project_evidence() and public_tool_update()."""
    shortened = []
    def clean(item: object, location: str) -> JSONValue:
        if isinstance(item, str):
            result = text(item)
            if result != item: shortened.append(location)
            return result
        if isinstance(item, list):
            if location.endswith('missing_fields'):
                return [clean(v,location) for v in item]
            if len(item)>3: shortened.append(location)
            return [clean(v,location) for v in item[:3]]
        if isinstance(item, dict):
            return {k:clean(v,location+'.'+k) for k,v in item.items()}
        return item

    preview = {k:value[k] for k in ('answer','coverage','identity','comparison_coverage','units',
                                   'quantitative_superiority_supported','causal_claim_supported','overlap_interpretation',
                                   'selection_complete','partial_reasons','missing_work','nrel_issue_codes','gaps','missing_comparison_fields')}
    if value.get('experimental_coverage') is not None:
        inventory = value['experimental_coverage']
        preview['experimental_coverage'] = {
            **inventory, 'enzyme': text(inventory['enzyme']),
            **{key: [{**c, 'form': text(c['form']), 'morphology': text(c['morphology'])}
                     for c in inventory[key]] for key in ('conditions', 'missing_in_both')}}
    # Workbook row references are scalar provenance labels, not numeric arrays.
    # Keep the outbound vector guard unchanged; never apply this to arbitrary arrays.
    preview['units'] = {**value['units'], 'record_examples': [
        {**example, 'workbook_rows': ', '.join(str(row) for row in example['workbook_rows'])}
        for example in value['units'].get('record_examples', [])[:3]]}
    preview['nrel'] = {k:value['nrel'][k] for k in ('total_matches','returned','truncated','unverified_filter_fields','selection')}
    preview['nrel']['observations'] = [{k:o[k] for k in ('enzyme_identifier','original_text','temperature_c','ph',
        'pet_form','source_row','source_column','unit_status','evidence_category','csv_sha256')} for o in value['nrel']['observations'][:3]]
    for card, original in zip(preview['nrel']['observations'], value['nrel']['observations']):
        if 'working_units' in original:
            card['working_units'] = original['working_units']
        if 'source_row_links' in original:
            card['source_row_links'] = original['source_row_links']
    preview['pazy'] = {k:value['pazy'][k] for k in ('total_matches','returned','truncated','matching_parent_records',
        'matching_protein_ids','matching_publications','total_matching_measurements','name_ambiguity','selection','native_identifier_policy','archive_sha256','schema_version')}
    cards = []
    condition_fields = ('temperature_c','ph','duration_value','duration_unit','enzyme_amount_value','enzyme_amount_unit',
        'enzyme_concentration_value','enzyme_concentration_unit','substrate_amount_value','substrate_amount_unit',
        'substrate_concentration_value','substrate_concentration_unit','reaction_volume_value','reaction_volume_unit',
        'enzyme_features','buffer','agitation_mode','agitation_speed_rpm')
    material_fields = ('form','morphology','crystallinity_percent','pretreatment','reported_name','supplier',
                       'thickness_value','thickness_unit','particle_size_value','particle_size_unit')
    for card in value['pazy']['records'][:3]:
        selected = {k:card[k] for k in ('parent_index','assay_index','protein_id','protein_name','publication_doi',
            'locator','evidence_locator','identity_status','total_measurements','omitted_measurements','out_of_scope_measurements','missing_fields')}
        selected['conditions'] = {k:card['context']['conditions'].get(k) for k in condition_fields}
        selected['material'] = {k:card['context']['material'].get(k) for k in material_fields}
        selected['detection_methods'] = card['context']['detection_methods'][:3]
        if len(card['context']['detection_methods'])>3: shortened.append(card['locator']+'.detection_methods')
        selected['measurements'] = [{k:m[k] for k in ('measurement_index','locator','record','missing_fields',
            'evidence_category','evidence_origin')} for m in card['measurements'][:3]]
        # All schema-valid measurement fields are small scalar descriptions, never arbitrary provider JSON.
        projected = clean(selected, card['locator'])
        projected['measurement_summary'] = measurement_summary(cast(AssaySummaryInput, projected))
        cards.append(projected)
    preview['pazy']['records'] = cards
    preview['comparisons'] = value['comparisons'][:8]
    # Preserve all scientific warnings, gap priorities and missing-field lists; only source strings/arrays above are bounded.
    preview['source_preview_omissions'] = shortened
    preview['context_projection'] = 'Whitelisted scalar protocol fields; source arrays at most 3, strings at most 300 characters. Additional context remains in the authorized original source.'
    return preview
