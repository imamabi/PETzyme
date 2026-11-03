"""Bounded transient source unions and exact-condition coverage, without pooling values."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import islice, product
import json
from typing import Callable, Literal, cast

from .agent_types import ResearchRequest
from .contracts import JSONValue
from .pet_presenter import AssaySummaryInput, measurement_summary

MAX_RUNS, MAX_NREL, MAX_ASSAYS, MAX_MEASUREMENTS, MAX_COMPARISONS = 8, 6, 6, 3, 16


def requested_condition_coverage(request: ResearchRequest, observations: list[dict], *,
                                 basis: Literal['NREL_CELLS', 'PAZY_ASSAYS'] = 'NREL_CELLS') -> dict[str, JSONValue]:
    """Check which requested exact conditions have actual NREL cells.

    Purpose: coverage basis for multi-study validation and display.
        Expands the request's identifier/temperature/pH/form dimensions
        (first 64 combos) and tests each against executed observations.
        PAZy identity or comparability never counts as NREL coverage.
    Returns:
        dict with basis, requested/covered counts, covered and missing
            condition lists, unlisted overflow, completeness flag, and
            unspecified fields.
    Flow: Receives ResearchRequest from interpret_request() and actual NREL observations from investigate_pet_evidence(); returns dict[str, JSONValue] for validate_investigation() and accumulate_pet_results()."""
    fields = ('enzyme_identifier', 'temperature_c', 'ph', 'pet_form')
    dimensions = [request[k] or [None] for k in ('enzyme_identifiers', 'temperatures', 'ph_values', 'pet_forms')]
    total = 1
    for values in dimensions: total *= len(values)
    targets = list(islice(product(*dimensions), 64))
    def found(target: tuple) -> bool:
        return any(all(expected is None or o.get(key) == expected for key, expected in zip(fields, target))
                   for o in observations)
    covered = [dict(zip(fields, target)) for target in targets if found(target)]
    missing = [dict(zip(fields, target)) for target in targets if not found(target)]
    return {'basis': ('Actual NREL cells; not PAZy sequence identity or experimental comparability.' if basis == 'NREL_CELLS' else
                      'Executed PAZy source-label/nominal-condition matches; not NREL cells, sequence identity or assay comparability.'),
            'requested': total, 'covered': len(covered), 'covered_conditions': covered,
            'missing_conditions': missing, 'unlisted_conditions': max(0, total - len(targets)),
            'complete': total == len(covered),
            'unspecified_fields': [field for field, values in zip(fields, dimensions) if values == [None]]}


def _union(values: list[list]) -> list:
    result = []
    for collection in values:
        for value in collection:
            if value not in result: result.append(deepcopy(value))
    return result


def _unique(items: list[dict], key: Callable[[dict], object]) -> tuple[list[dict], int]:
    """Collapse identical source-bound objects; retain and flag contradictory representations."""
    groups, result, conflicts = {}, [], 0
    for item in items:
        identity = key(item)
        content = json.dumps(item, sort_keys=True)
        prior = groups.setdefault(identity, set())
        if content in prior: continue
        if prior: conflicts += 1
        prior.add(content)
        result.append(deepcopy(item))
    return result, conflicts


def accumulate_pet_results(runs: list[dict], request: ResearchRequest, additional_nrel: list[dict] | None = None) -> dict[str, JSONValue]:
    """Union 1–8 executed multi-study previews into one transient display object.

    Purpose: merges NREL observations and PAZy assay cards across runs
        for render_multi_study, deduplicating identical source-bound
        objects while flagging contradictory representations as
        conflicts. Refuses to mix different archive pins or NREL file
        versions in one union.
    Raises:
        ValueError: on empty/oversize run lists or mixed source versions.
    Flow: Receives bounded call_id/preview dict list from present_investigation() and ResearchRequest from interpret_request(); returns transient dict[str, JSONValue] for render_multi_study()."""
    if not runs or len(runs) > MAX_RUNS: raise ValueError('Supply one to eight executed multi-study previews')
    previews = [r['preview'] for r in runs]
    pins = {(p['pazy']['archive_sha256'], p['pazy']['schema_version']) for p in previews}
    if len(pins) != 1: raise ValueError('Different activity sources cannot share a presentation union')
    if len({o['csv_sha256'] for p in previews for o in p['nrel']['observations']}) > 1:
        raise ValueError('Different NREL versions cannot share a presentation union')
    value = deepcopy(previews[0])
    value['experimental_coverages'] = _union([
        [p['experimental_coverage']] if p.get('experimental_coverage') else [] for p in previews])
    unit_examples = _union([p['units'].get('record_examples', []) for p in previews])
    value['units']['record_examples'] = unit_examples[:MAX_NREL]
    value['units']['record_examples_omitted'] = max(0, len(unit_examples) - MAX_NREL)
    all_nrel = [o for p in previews for o in p['nrel']['observations']]
    if additional_nrel:
        keys = {'enzyme_identifier','original_text','temperature_c','ph','pet_form','source_row','source_column',
                'unit_status','evidence_category','csv_sha256','working_units','source_row_links'}
        all_nrel.extend({k:v for k,v in o.items() if k in keys} for o in additional_nrel)
        if len({o['csv_sha256'] for o in all_nrel}) > 1:
            raise ValueError('Different NREL versions cannot share a presentation union')
    # Documentary links depend on each call's PAZy preview. Merge those links;
    # differing original measurements/units still remain genuine conflicts.
    nrel, nrel_conflicts = _unique([{k:v for k,v in o.items() if k != 'source_row_links'} for o in all_nrel],
                                 lambda o:(o['csv_sha256'],o['source_row'],o['source_column']))
    for observation in nrel:
        observation['source_row_links'] = _union([o.get('source_row_links', []) for o in all_nrel
            if {k:v for k,v in o.items() if k != 'source_row_links'} == observation])
    # Group assays by archive-qualified pointer. Keep distinct scalar/construct contexts separate.
    cards, by_identity, conflicts = [], {}, nrel_conflicts
    for p in previews:
        for card in p['pazy']['records']:
            key = (p['pazy']['archive_sha256'], card['locator'])
            context = {k:v for k,v in card.items() if k not in {
                'measurements','omitted_measurements','out_of_scope_measurements','identity_status','measurement_summary'}}
            encoded = json.dumps(context,sort_keys=True)
            candidates = by_identity.setdefault(key, {})
            if encoded not in candidates:
                if candidates: conflicts += 1
                merged = deepcopy(card); merged['measurements'] = []
                candidates[encoded] = merged; cards.append(merged)
            merged = candidates[encoded]
            merged['measurements'].extend(deepcopy(card['measurements']))
    measurement_keys, hidden_measurements = set(), 0
    for card in cards:
        measurements, count = _unique(card['measurements'], lambda m:m['locator'])
        conflicts += count
        measurement_keys.update((card['locator'],m['locator']) for m in measurements)
        hidden_measurements += max(0,len(measurements)-MAX_MEASUREMENTS)
        card['measurements'] = measurements[:MAX_MEASUREMENTS]
        card['omitted_measurements'] = card['out_of_scope_measurements'] = None
        card['identity_status'] = 'PER_CALL_ASSESSMENTS'
        card['measurement_summary'] = measurement_summary(cast(AssaySummaryInput, card))
    comparisons, count = _unique([c for p in previews for c in p['comparisons']],
        lambda c:(c['nrel_record'],c['nrel_column'],c['pazy_locator'],c['publication_doi']))
    call_coverage = [{'call_id':r['call_id'], 'nrel':{k:r['preview']['nrel'][k] for k in ('returned','total_matches','truncated','unverified_filter_fields')},
                     'pazy':{k:r['preview']['pazy'][k] for k in ('returned','total_matches','total_matching_measurements','truncated')},
                     'partial_reasons':r['preview']['partial_reasons'],
                     'assays':[{'locator':c['locator'], 'returned_measurements':len(c['measurements']),
                                'omitted_measurements':c['omitted_measurements'],
                                'out_of_scope_measurements':c['out_of_scope_measurements']}
                               for c in r['preview']['pazy']['records']],
                     'conditions':requested_condition_coverage(request,r['preview']['nrel']['observations']),
                     'identity':r['preview']['identity']} for r in runs]
    value['accumulation'] = {
        'run_count':len(runs), 'calls':call_coverage,
        'condition_coverage':requested_condition_coverage(request,nrel),
        'unique_nrel_retrieved':len({(o['csv_sha256'],o['source_row'],o['source_column']) for o in nrel}),
        'unique_pazy_assays_retrieved':len({c['locator'] for c in cards}),
        'unique_pazy_measurements_retrieved':len(measurement_keys),
        'unique_comparisons_retrieved':len(comparisons), 'source_conflicts':conflicts,
        'display_omitted_nrel':max(0,len(nrel)-MAX_NREL),
        'display_omitted_assays':max(0,len(cards)-MAX_ASSAYS),
        'display_omitted_measurements':hidden_measurements + sum(len(c['measurements']) for c in cards[MAX_ASSAYS:]),
        'display_omitted_comparisons':max(0,len(comparisons)-MAX_COMPARISONS),
        'interpretation':'Unique source evidence retrieved, not independent experiments; query match totals are per-call and are never summed.'}
    if len(runs) == 1 and not additional_nrel:
        return value
    value['nrel'].update(observations=nrel[:MAX_NREL],returned=min(len(nrel),MAX_NREL),total_matches=None,
                         truncated=any(p['nrel']['truncated'] for p in previews) or len(nrel)>MAX_NREL,
                         unverified_filter_fields=_union([p['nrel']['unverified_filter_fields'] for p in previews]))
    value['pazy'].update(records=cards[:MAX_ASSAYS],returned=min(len(cards),MAX_ASSAYS),total_matches=None,
                         total_matching_measurements=None, matching_parent_records=None,matching_publications=None,
                         matching_protein_ids=None,truncated=any(p['pazy']['truncated'] for p in previews) or len(cards)>MAX_ASSAYS or hidden_measurements>0)
    value['comparisons'] = comparisons[:MAX_COMPARISONS]
    value['comparison_coverage'] = {'candidate_pairs_in_returned_preview':None,'returned':len(value['comparisons']),
                                  'truncated':any(p['comparison_coverage']['truncated'] for p in previews) or len(comparisons)>MAX_COMPARISONS,
                                  'classifications':dict(Counter(c['category'] for c in value['comparisons']))}
    value['identity'] = {'verified':None,'status':'PER_CALL_ASSESSMENTS','historical_checks_reused':False}
    for key in ('partial_reasons','missing_work','nrel_issue_codes','gaps','missing_comparison_fields','source_preview_omissions'):
        value[key] = _union([p[key] for p in previews])
    value['selection_complete'] = all(p['selection_complete'] for p in previews) and not conflicts and not any(
        value['accumulation'][k] for k in ('display_omitted_nrel','display_omitted_assays','display_omitted_measurements','display_omitted_comparisons'))
    value['answer'] = (f"Retrieved {value['accumulation']['unique_nrel_retrieved']} distinct NREL cells and "
                       f"{value['accumulation']['unique_pazy_assays_retrieved']} distinct PAZy assay projections across executed source queries. "
                       'Values remain separate; this union does not establish activity superiority or causality, and identity assessments remain call-bound.')
    return value
