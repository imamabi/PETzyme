"""Local multi-study composition; no network, persistence, prediction or response pooling."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from .activity_snapshot import iter_assays
from .activity_types import ProteinVerification
from .contracts import JSONValue, Observation, ToolResult
from .cross_comparability import (
    assess_cross_source_comparability,
    describe_nrel_observation,
    describe_pazy_assay,
)
from .cross_overlap import NREL_PAPER, candidate_protein_pairs, identify_cross_source_overlap
from .evidence import assess_evidence
from .experimental_coverage import summarize_experimental_coverage
from .nrel_units import standardized_nrel_values, workbook_reference_link
from .observations import filter_observations, load_nrel, parse_nrel
from .pet_retrieval import (
    ROOT,
    EvidenceQuery,
    diverse_assays,
    load_activity_source,
    match_pet_assays,
    project_pet_assays,
    valid_evidence_query,
)
from .results import outcome

LIMITATIONS = [
    'PAZy is publication-derived curation, not independent replication; shared DOI or workbook references must not multiply experimental evidence.',
    'Names and aliases are UNVERIFIED identity candidates. Historical aggregate sequence checks are not reusable record-level identity proofs.',
    'Original NREL CSV unit declarations remain UNKNOWN/UNVERIFIED. Validated cells have a separately labelled COMPUTED ×1000 representation based on INFERRED mmol/mg units; no cross-source pooling or ranking.',
    'Missing metadata is not proof that a procedure was absent. A 48-hour product-yield endpoint is not a kinetic rate or kcat.',
    'Only bounded source-attributed previews may be displayed or forwarded; PAZy records and derived selections are transient, without exports or session history.',
]


def _nrel_scope(observations: list[Observation], names: set[str] | None, query: EvidenceQuery) -> list[Observation]:
    """Use the existing exact NREL filter while keeping undocumented condition fields explicit elsewhere."""
    forms = {'powder':'CRYSTALLINE_POWDER', 'film':'AMORPHOUS_FILM'}
    if 'form' in query and query['form'] not in forms: return []
    selected = filter_observations(observations, temperature_c=query.get('temperature_c'), ph=query.get('ph'),
                                   pet_form=forms.get(query.get('form')))['result'] or []
    return [o for o in selected if (names is None or o['enzyme_identifier'] in names)
            and ('morphology' not in query or query['morphology'] ==
                 ('crystalline' if o['pet_form'] == 'CRYSTALLINE_POWDER' else 'amorphous'))]


def investigate_pet_evidence(query: EvidenceQuery | None = None, limit: int = 3,
                             scope: Literal['ALL', 'OUTSIDE_NREL_PUBLICATION'] = 'ALL',
                             verification: ProteinVerification | None = None,
                             source_scope: Literal['BOTH', 'PAZY_ONLY'] = 'BOTH') -> ToolResult[dict[str, JSONValue]]:
    """Investigate PET evidence across NREL and the pinned PAZy curation.

    Purpose: the bounded local multi-study composition. Loads both pinned
        sources itself, selects diverse PAZy assay previews plus
        distinct-condition NREL references, runs cross-source
        comparability on name-candidate pairs (at most 8 shown), and
        reports archive-wide coverage, unit provenance, fixed verification
        gaps, and explicit partial reasons. No pooling, ranking, unit
        equivalence, or causal claim is ever produced. Explicit inferred unit representations remain separate from original observations.
    Args:
        query: EvidenceQuery filters (None matches everything).
        limit: 1..3 assays/references in the preview.
        source_scope: BOTH sources, or PAZY_ONLY without opening NREL data.
        scope: 'ALL' or 'OUTSIDE_NREL_PUBLICATION' (excludes assays from
            the NREL paper's DOI for independent-context questions).
        verification: optional bound ProteinVerification from an actually
            executed compare_identity; must reference a snapshot
            candidate pair and this NREL source or the call fails.
    Returns:
        ToolResult[dict] with 'answer', 'coverage', 'nrel', 'pazy',
            'identity', 'comparisons', 'gaps', 'units',
            'quantitative_superiority_supported' (always False),
            'causal_claim_supported' (always False), 'partial_reasons',
            and 'persistence' (TRANSIENT_PAZY_SELECTION).
    Statuses:
        SUCCESS when nothing truncated; PARTIAL with explicit reasons
            otherwise; ERROR on bad inputs, unavailable sources, missing
            unit audit, or an unbound identity proof.
    Example:
        >>> out = investigate_pet_evidence({'protein_name': 'LCC-ICCG'}, limit=1)
        >>> (out['status'] in ('SUCCESS', 'PARTIAL'), out['result']['persistence'])
        (True, 'TRANSIENT_PAZY_SELECTION')
    Limitations:
        Same-publication curation is a potential duplicate, never
            independent replication; historical identity checks are never
            reused as proofs.
    Flow: Receives external EvidenceQuery/limit/scope and optional bound ProteinVerification from verify_snapshot_identities() or execute_tool_call(); returns ToolResult[dict[str, JSONValue]] for project_evidence() and the transient multi-study interface."""
    query = {} if query is None else query
    if not valid_evidence_query(query) or type(limit) is not int or not 1 <= limit <= 3 or scope not in ('ALL','OUTSIDE_NREL_PUBLICATION') or source_scope not in ('BOTH', 'PAZY_ONLY'):
        return outcome('ERROR', None, [], LIMITATIONS, 'COMPUTED', 'Use exact documented filters, scope and a limit of 1–3.')
    loaded = load_activity_source()
    nrel = load_nrel(str(ROOT / 'data/nrel/label_data.csv')) if source_scope == 'BOTH' else None
    if loaded['result'] is None or nrel is not None and nrel['result'] is None:
        return outcome('ERROR', None, loaded['provenance'] + (nrel['provenance'] if nrel else []), LIMITATIONS, 'COMPUTED',
                       'Required pinned local source unavailable; no download, live lookup or fallback was attempted.')
    snapshot = loaded['result']
    table = nrel['result'] if nrel else None
    original = []
    if table is not None:
        parsed = parse_nrel(table)
        if parsed['result'] is None:
            return outcome('ERROR', None, parsed['provenance'], LIMITATIONS, 'COMPUTED', 'Original NREL observations could not be parsed.')
        original = parsed['result']['observations']
    matches = match_pet_assays(snapshot, query)
    if scope == 'OUTSIDE_NREL_PUBLICATION':
        matches = [r for r in matches if r['publication_doi'].casefold() != NREL_PAPER]
    retrieved = project_pet_assays(snapshot, matches, query, limit)
    selected_refs = diverse_assays(matches, limit)
    pairs = candidate_protein_pairs(snapshot, table) if table is not None else []
    if verification is not None:
        try:
            valid_proof = (table is not None and (verification['pazy_protein_id'], verification['nrel_identifier']) in pairs
                and verification['identity'] in ('EXACT_SEQUENCE_MATCH','DIFFERENT_SEQUENCE','INSUFFICIENT_INFORMATION')
                and any(s['name']=='NREL' and s['url']==table['source']['url'] and s['locator']
                    and Path(s['locator']).resolve()==Path(table['source']['locator']).resolve() for s in verification['provenance']))
        except (KeyError, TypeError, ValueError, OSError):
            valid_proof = False
        if not valid_proof:
            return outcome('ERROR', None, [], LIMITATIONS, 'COMPUTED', 'Identity proof is not bound to this NREL source and a snapshot candidate.')
    if verification:
        for card in retrieved['result']['records']:
            if card['protein_id'] == verification['pazy_protein_id']: card['identity_status'] = verification['identity']
    candidate_map = {}
    for pid, name in pairs: candidate_map.setdefault(pid, set()).add(name)
    names = None
    if 'protein_name' in query:
        # A supplied exact NREL name remains a reference even when PAZy has no such label.
        names = {query['protein_name']} | {name for r in matches for name in candidate_map.get(r['protein_id'], set())}
    elif any(k in query for k in ('protein_id','parent_index','assay_index')):
        names = {name for r in matches for name in candidate_map.get(r['protein_id'], set())}
    nrel_matches = _nrel_scope(original, names, query)
    # Inventory all conditions for the requested label, retaining publication/protocol scope.
    # It never substitutes these counts for the bounded exact response selection below.
    condition_inventory = None
    if 'protein_name' in query:
        inventory_query = {k: v for k, v in query.items()
                           if k not in ('temperature_c', 'ph', 'form', 'morphology', 'measurement_index')}
        inventory_refs = match_pet_assays(snapshot, inventory_query)
        if scope == 'OUTSIDE_NREL_PUBLICATION':
            inventory_refs = [r for r in inventory_refs if r['publication_doi'].casefold() != NREL_PAPER]
        inventory_names = {query['protein_name']} | {
            name for r in inventory_refs for name in candidate_map.get(r['protein_id'], set())}
        condition_inventory = summarize_experimental_coverage(
            query['protein_name'], [o for o in original if o['enzyme_identifier'] in inventory_names],
            inventory_refs, source_scope=source_scope)
    # Prefer distinct conditions for the bounded reference selection, never response magnitude.
    selected_nrel, seen = [], set()
    for o in nrel_matches:
        key = (o['temperature_c'], o['ph'], o['pet_form'])
        if key not in seen: selected_nrel.append(o); seen.add(key)
        if len(selected_nrel) == limit: break
    comparisons = []
    total_pairs = 0
    for ref, card in zip(selected_refs, retrieved['result']['records']):
        for o in selected_nrel:
            # Even a name candidate is never upgraded to verified identity here.
            if (ref['protein_id'], o['enzyme_identifier']) not in pairs: continue
            for m in card['measurements']:
                total_pairs += 1
                # The preview shows 8 pairs but counts all of them, so truncation stays explicit.
                if len(comparisons) >= 8: continue
                assessed = assess_cross_source_comparability(describe_nrel_observation(o),
                    describe_pazy_assay(ref, m['measurement_index']), verification)['result']
                comparisons.append({**assessed, 'nrel_record': o['source_row'], 'nrel_column': o['source_column'],
                    'pazy_locator': m['locator'], 'publication_doi': ref['publication_doi'],
                    'nominal_temperature_ph_overlap': o['temperature_c'] == ref['assay']['conditions'].get('temperature_c')
                        and o['ph'] == ref['assay']['conditions'].get('ph')})
    # The established audit computes source-wide overlap; no previous live proofs are fabricated.
    audited = identify_cross_source_overlap(snapshot, table, [verification] if verification else None)['result'] if table is not None else None
    global_refs = list(iter_assays(snapshot))
    outside = [r for r in global_refs if r['publication_doi'].casefold() != NREL_PAPER]
    cells = {(o['enzyme_identifier'],o['temperature_c'],o['ph'], 'powder' if o['pet_form']=='CRYSTALLINE_POWDER' else 'film') for o in original}
    same_study_missing = sum(r['publication_doi'].casefold() == NREL_PAPER and
        not any((name,r['assay']['conditions'].get('temperature_c'),r['assay']['conditions'].get('ph'),r['assay']['material'].get('form')) in cells
                for name in candidate_map.get(r['protein_id'],set())) for r in global_refs)
    coverage = {k: audited[k] for k in ('candidate_pair_count','name_condition_pairs','comparison_categories',
                                       'target_publication_assays','verified_candidate_pairs')} if audited else {'cross_source_overlap': 'NOT_ASSESSED; NREL excluded by source scope'}
    coverage.update(pet_assays=len(global_refs), pet_publications=len({r['publication_doi'].casefold() for r in global_refs}),
                    outside_publication_assays=len(outside), outside_publications=len({r['publication_doi'].casefold() for r in outside}),
                    same_study_assays_without_condensed_cell=same_study_missing,
                    condition_overlap_rule='Name/alias candidates plus exact temperature/pH/form; not authenticated original-experiment equivalence.',
                    scope='FULL_PINNED_ARCHIVE; classifications use name/nominal-condition pairs and only explicitly supplied bound identity proofs')
    unknown_filters = sorted(set(query)-{'protein_name','temperature_c','ph','form','morphology','protein_id','parent_index','assay_index','measurement_index'})
    nrel_cards = [{k:o[k] for k in ('enzyme_identifier','original_text','temperature_c','ph','pet_form','source_row',
                                    'source_column','unit_status','evidence_category','csv_sha256')} for o in selected_nrel]
    for card, o, representation in zip(nrel_cards, selected_nrel, standardized_nrel_values(selected_nrel)):
        card['source'] = {**o['source'], 'locator': f"#record={o['source_row']}&column={o['source_column']}"}
        card['working_units'] = representation
        card['source_row_links'] = [
            {'pazy_assay': p['locator'], 'status': workbook_reference_link(
                representation['workbook_rows'], p['publication_doi'], p['evidence_locator'])}
            for p in retrieved['result']['records']
            if p['context']['conditions'].get('temperature_c') == o['temperature_c']
            and p['context']['conditions'].get('ph') == o['ph']]
    missing_fields = sorted({f for c in comparisons for f in c['missing_fields']})
    gaps = [
        {'priority':1,'code':'VERIFY_PROTEIN_AND_CONSTRUCT','requirement':'Verify exact parent sequences using the existing authoritative live identity mechanism, then verify the assayed construct/variant. Name equality alone is insufficient.'},
        {'priority':2,'code':'AUTHENTICATE_CSV_UNITS','requirement':'Obtain the authors’ explicit condensed-CSV unit, rounding and aggregation declaration; retain every original cell.'},
        {'priority':3,'code':'LINK_ORIGINAL_EXPERIMENT','requirement':'Authenticate original study/experiment and source-row mappings before identifying independent replication or duplicate representations.'},
        {'priority':4,'code':'RECONCILE_PROTOCOL','requirement':'Verify substrate form/morphology, duration, loadings, detection/analyte definition, units, replicates and uncertainty under matched conditions; then reassess comparability.'},
    ]
    if source_scope == 'PAZY_ONLY':
        gaps = [g for g in gaps if g['code'] != 'AUTHENTICATE_CSV_UNITS']
    units = {'csv_units': 'NREL excluded by explicit source scope', 'pazy_units': 'Original reported units retained without conversion.'}
    if table is not None:
        try:
            audit = json.loads((ROOT/'NREL_SOURCE_MANIFEST.json').read_text())['unit_reconciliation']
            if audit['csv_sha256'] != table['csv_sha256']: raise ValueError('NREL unit audit belongs to a different CSV')
            units = {'workbook_units':'VERIFIED in the prior source-workbook audit; µmol aromatic products/mg enzyme after 48 h, not a rate.',
                     'csv_generation':'INFERRED rounded aggregation; validated source/cell certificate from the explicit in-memory workbook audit. Runtime investigations do not download the workbook.',
                     'csv_units':'UNRESOLVED original declaration (UNKNOWN/UNVERIFIED); supported working mmol/mg interpretation is INFERRED, with a separate COMPUTED µmol/mg representation.', 'audit_evidence_category':'COMPUTED',
                     'source_doi':audit['supplement_doi'], 'source_locator':audit['paper_unit_locator']}
            units['record_examples'] = [
                {k: example[k] for k in ('csv_record', 'column', 'original', 'workbook_rows',
                                         'reconstructed_mmol_scale_mean', 'direct_W_mean_umol_per_mg',
                                         'hypothetical_CSV_times_1000')}
                for example in audit.get('selected_ESM050_matches', [])
                if any(o['source_row'] == example['csv_record'] and o['source_column'] == example['column']
                       and o['original_text'] == example['original'] for o in selected_nrel)]
            units['record_interpretation'] = (
                'These checksum-bound prior workbook audit examples are COMPUTED diagnostics, not converted observations. '
                'The rounded aggregation hypothesis can reproduce a CSV cell while CSV ×1000 differs from the '
                'unrounded workbook mean. An assay citing the same workbook rows supplies a linkage candidate; '
                'it does not authenticate the CSV generating procedure, sequence or experimental construct.')
        except (OSError, KeyError, ValueError, TypeError):
            return outcome('ERROR', None, [], LIMITATIONS, 'COMPUTED', 'Unit provenance unavailable; no replacement interpretation generated.')
        working = json.loads((ROOT/'NREL_SOURCE_MANIFEST.json').read_text()).get('working_unit_interpretation', {})
        units['working_interpretation'] = working.get('interpretation', 'No adopted working inference available.')
        units['working_validation_counts'] = working.get('counts', {})
        units['lcc_rounding_diagnostics'] = working.get('lcc_rounding_diagnostics', [])
    partial_reasons, missing_work = [], []
    if retrieved['status']=='PARTIAL':
        partial_reasons.append('PAZY_PREVIEW_TRUNCATED')
        missing_work.append('Inspect omitted matching PAZy assays or measurements under a scoped follow-up.')
    if len(nrel_matches)>len(nrel_cards):
        partial_reasons.append('NREL_PREVIEW_TRUNCATED')
        missing_work.append('Inspect omitted NREL reference observations under a scoped follow-up.')
    if total_pairs>len(comparisons):
        partial_reasons.append('COMPARISON_PREVIEW_TRUNCATED')
        missing_work.append('Inspect omitted candidate comparison pairs in the returned preview scope.')
    if unknown_filters and nrel_matches:
        partial_reasons.append('NREL_FILTER_UNVERIFIED')
        missing_work.append('Authenticate the requested publication/protocol filters for the NREL references; their applicability is not established by the CSV.')
    partial = bool(partial_reasons)
    answer = (f"Selected {len(nrel_cards)} of {len(nrel_matches)} original NREL reference observations and "
              f"{retrieved['result']['returned']} of {retrieved['result']['total_matches']} matching PAZy PET assays. "
              'These sources do not establish activity superiority or a causal effect; identity and experimental equivalence remain unresolved.')
    if condition_inventory is not None:
        answer += (f" NREL contributes {condition_inventory['nrel_observations']} reference observations; PAZy contains {condition_inventory['pazy_assays']} assays and {condition_inventory['pazy_measurements']} source-reported measurements. "
                   + (f"PAZy adds {condition_inventory['pazy_additional_conditions']} nominal condition combinations, not independent replication." if source_scope == 'BOTH' else 'NREL was excluded; cross-source additions were not assessed.'))
    if not matches: answer += ' No PAZy assay matched these exact filters; this does not establish absence of evidence or sequence overlap.'
    evidence = assess_evidence(selected_nrel)['result']
    result = {'source_scope': source_scope, 'answer':answer,'coverage':coverage, 'query':query, 'scope':scope,
              'experimental_coverage':condition_inventory,
              'nrel':{'observations':nrel_cards,'total_matches':len(nrel_matches),'returned':len(nrel_cards),
                      'truncated':len(nrel_matches)>len(nrel_cards),'unverified_filter_fields':unknown_filters,
                      'selection':'Source-order distinct conditions; NREL references do not authenticate PAZy publication/protocol filters.'},
              'pazy':retrieved['result'], 'identity':{'verified':len({c['protein_id'] for c in retrieved['result']['records'] if c['identity_status']=='EXACT_SEQUENCE_MATCH'}),
                  'status':verification['identity'] if verification and any(c['protein_id']==verification['pazy_protein_id'] for c in retrieved['result']['records']) else 'UNVERIFIED',
                  'historical_checks_reused':False},
              'comparisons':comparisons, 'comparison_coverage':{'candidate_pairs_in_returned_preview':total_pairs,'returned':len(comparisons),
                  'truncated':total_pairs>len(comparisons),'classifications':dict(Counter(c['category'] for c in comparisons))},
              'overlap_interpretation':'Same-study PAZy workbook curation is a potential duplicate representation, never independent experimental replication. Shared DOI alone does not identify a specific experiment.',
              'missing_comparison_fields':missing_fields,'nrel_issue_codes':[i['code'] for i in evidence['issues']],
              'gaps':gaps,'units':units,'quantitative_superiority_supported':False,'causal_claim_supported':False,
              'selection_complete':not partial, 'partial_reasons':partial_reasons, 'missing_work':missing_work,
              'persistence':'TRANSIENT_PAZY_SELECTION'}
    return outcome('PARTIAL' if partial else 'SUCCESS', result, ([table['source']] if table is not None else []) + [snapshot['source']],
                   [*LIMITATIONS,*retrieved['limitations']], 'COMPUTED')
