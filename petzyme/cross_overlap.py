"""Audit candidate/publication/condition overlap; live sequence verifications stay transient."""
from __future__ import annotations

from collections import Counter, defaultdict

from .activity_snapshot import LIMITATIONS, iter_assays
from .activity_types import ActivitySnapshot, ProteinVerification
from .contracts import JSONValue, NrelTable, ToolResult
from .cross_comparability import assess_cross_source_comparability, describe_nrel_observation, describe_pazy_assay
from .identity import compare_identity
from .metadata import get_pazy_metadata
from .observations import parse_nrel, select_nrel_protein
from .pazy import PazyClient
from .results import outcome

NREL_PAPER = '10.1021/acscatal.5c03460'


def candidate_protein_pairs(snapshot: ActivitySnapshot, nrel: NrelTable) -> list[tuple[int, str]]:
    """List (pazy_protein_id, nrel_name) pairs sharing a PET-record name/alias.

    Purpose: finds every snapshot protein whose documented name or alias
        exactly equals an NREL label (PET-substrate records only).
        Output is candidate scope for verification, never identity proof.
    Returns:
        Sorted list of candidate tuples; empty when no names coincide.
    Flow: Receives ActivitySnapshot from load_pazy_activity_snapshot() and NrelTable from load_nrel(); returns name/alias candidate list[tuple[int, str]] for verify_snapshot_identities() and identify_cross_source_overlap(), never identity proof."""
    identifiers = {row[nrel['columns'].index('')] for row in nrel['rows']}
    candidates = set()
    for record in snapshot['records']:
        if not any(a['substrate'] == 'PET' for a in record['assays']): continue
        names = [record['protein_name'], *(s.get('name') for s in record.get('synonyms') or [])]
        candidates.update((record['protein_id'], name) for name in names if name in identifiers)
    return sorted(candidates)


def verify_snapshot_identities(snapshot: ActivitySnapshot, nrel: NrelTable, candidates: list[tuple[int, str]],
                                *, client: PazyClient | None = None) -> ToolResult[list[ProteinVerification]]:
    """Verify 1–3 name candidates by live PAZy lookup plus exact comparison.

    Purpose: for each snapshot-derived pair, resolves the NREL row,
        fetches the live PAZy record, and runs compare_identity. Stops
        at the first failure (PARTIAL) with no retries. Results are
        transient: only aggregate outcomes may be persisted, never
        records, sequences, or hashes.
    Args:
        snapshot: ActivitySnapshot scoping allowed candidates.
        nrel: NrelTable for local resolution.
        candidates: 1..3 distinct pairs from candidate_protein_pairs().
        client: optional injected PazyClient for tests.
    Returns:
        ToolResult[list[ProteinVerification]] with per-pair identity and
            provenance.
    Statuses:
        SUCCESS when all checked; PARTIAL on first failure; ERROR when
            the candidate list is invalid.
    Flow: Receives at most three snapshot-derived candidate pairs from candidate_protein_pairs(); returns transient ToolResult[list[ProteinVerification]] from get_pazy_metadata()/compare_identity() for identify_cross_source_overlap()."""
    allowed = set(candidate_protein_pairs(snapshot, nrel))
    if not 1 <= len(candidates) <= 3 or len(set(candidates)) != len(candidates) or any(c not in allowed for c in candidates):
        return outcome('ERROR', None, [], LIMITATIONS, 'COMPUTED', 'Select 1–3 distinct snapshot-derived name candidates; none were requested.')
    checks, sources = [], []
    for protein_id, name in candidates:
        local = select_nrel_protein(nrel, name)
        if local['result'] is None:
            return outcome('PARTIAL', checks, sources, LIMITATIONS, 'COMPUTED', 'Local candidate unavailable; remaining checks not attempted.')
        live = get_pazy_metadata(protein_id, client=client)
        if live['result'] is None:
            return outcome('PARTIAL', checks, sources, LIMITATIONS, 'COMPUTED', 'Live lookup failed; remaining checks not attempted. No retry or snapshot protein fallback.')
        result = compare_identity(local['result'], live['result'])
        checks.append({'pazy_protein_id': protein_id, 'nrel_identifier': name,
                       'identity': result['result']['identity'], 'provenance': result['provenance']})
        sources.extend(s for s in result['provenance'] if s not in sources)
    return outcome('SUCCESS', checks, sources, [*LIMITATIONS,
        'Live verification details/provenance are transient; persist only aggregate outcomes, never API records, sequences or sequence hashes.',
        'The candidate subset is not a complete sequence search; unexamined proteins remain unresolved.'], 'COMPUTED')


def identify_cross_source_overlap(snapshot: ActivitySnapshot, nrel: NrelTable,
                                   verifications: list[ProteinVerification] | None = None) -> ToolResult[dict[str, JSONValue]]:
    """Audit archive-wide candidate/publication/condition overlap (aggregates only).

    Purpose: Build 07A audit composition. Classifies every candidate
        pair's nominal comparability, counts candidate/name-condition
        pairs and target-publication assays, and accepts only bound,
        deduplicated identity proofs. Emits aggregate counts for the
        audit report — deliberately not registered as an agent tool.
    Args:
        snapshot: ActivitySnapshot from load_pazy_activity_snapshot().
        nrel: NrelTable from load_nrel().
        verifications: optional transient proofs from
            verify_snapshot_identities(); unbound/duplicated proofs fail.
    Returns:
        ToolResult[dict] of aggregate overlap/coverage counts.
    Statuses:
        SUCCESS normally; ERROR on unavailable NREL data or unbound proofs.
    Flow: Receives ActivitySnapshot/NrelTable from their loaders and optional transient verify_snapshot_identities() results; returns ToolResult[dict[str, JSONValue]] for audit_build07a.main() aggregate reporting only."""
    parsed = parse_nrel(nrel)
    if parsed['result'] is None:
        return outcome('ERROR', None, [nrel['source'], snapshot['source']], LIMITATIONS, 'COMPUTED', 'NREL observations unavailable')
    candidates = candidate_protein_pairs(snapshot, nrel)
    allowed = set(candidates)
    proof_index = {}
    for proof in verifications or []:
        key = proof['pazy_protein_id'], proof['nrel_identifier']
        if (key not in allowed or key in proof_index or not proof['provenance']
                or proof['identity'] not in ('EXACT_SEQUENCE_MATCH', 'DIFFERENT_SEQUENCE', 'INSUFFICIENT_INFORMATION')):
            return outcome('ERROR', None, [], LIMITATIONS, 'COMPUTED', 'Identity proof is duplicated or not bound to a snapshot candidate')
        proof_index[key] = proof
    by_protein = defaultdict(list)
    for observation in parsed['result']['observations']:
        by_protein[observation['enzyme_identifier']].append(observation)
    by_pazy = defaultdict(list)
    for protein_id, identifier in candidates: by_pazy[protein_id].append(identifier)
    selected = list(iter_assays(snapshot))
    nominal_global = {(o['temperature_c'], o['ph']) for o in parsed['result']['observations']}
    classifications = Counter({key: 0 for key in ('VERIFIED_COMPARABLE','LIMITED_COMPARISON','NOT_COMPARABLE','INSUFFICIENT_INFORMATION')})
    missing, incompatible, identity_counts = Counter(), Counter(), Counter()
    matched_cells, matched_assays, candidate_assays, pairs = set(), set(), set(), 0
    no_nominal = 0
    documented_duplicates = 0
    for reference in selected:
        assay = reference['assay']; conditions = assay['conditions']
        descriptor = None
        for name in by_pazy[reference['protein_id']]:
            candidate_assays.add((reference['parent_index'], reference['assay_index']))
            hits = [o for o in by_protein[name] if (o['temperature_c'], o['ph']) ==
                    (conditions.get('temperature_c'), conditions.get('ph'))]
            no_nominal += int(not hits)
            if descriptor is None:
                measurements = assay['measurements']
                index = next((i for i,m in enumerate(measurements) if m.get('metric_type') == 'specific_activity'), 0)
                descriptor = describe_pazy_assay(reference, index)
            for observation in hits:
                pairs += 1
                matched_cells.add((observation['source_row'], observation['source_column']))
                matched_assays.add((reference['parent_index'], reference['assay_index']))
                comparison = assess_cross_source_comparability(describe_nrel_observation(observation), descriptor,
                                                                proof_index.get((reference['protein_id'], name)))['result']
                classifications[comparison['category']] += 1
                documented_duplicates += int(comparison['duplicate_status'] == 'SAME_DOCUMENTED_EXPERIMENT')
                missing.update(comparison['missing_fields']); incompatible.update(comparison['incompatible_fields'])
                identity_counts[comparison['identity_status']] += 1
    targeted = [r for r in snapshot['records'] if r['literature_doi'].casefold() == NREL_PAPER]
    ambiguities = sum(len(set(names)) > 1 for names in by_pazy.values())
    result = {
        'persistence': 'TRANSIENT_LIVE_IDENTITY_DETAILS' if verifications else 'SNAPSHOT_AUDIT_ONLY',
        'nrel_records': len(nrel['rows']), 'nrel_observations': len(parsed['result']['observations']),
        'pet_snapshot_proteins': len({r['protein_id'] for r in selected}),
        'candidate_pairs': [{'pazy_snapshot_protein_id': pid, 'nrel_identifier': name,
                             'identity': proof_index[(pid,name)]['identity'] if (pid,name) in proof_index else 'NAME_OR_ALIAS_ONLY'}
                            for pid,name in candidates],
        'candidate_pair_count': len(candidates), 'ambiguous_name_candidate_proteins': ambiguities,
        'verified_candidate_pairs': len(proof_index),
        'verification_outcomes': dict(Counter(p['identity'] for p in proof_index.values())),
        'unexamined_candidate_pairs': len(candidates) - len(proof_index),
        'target_publication': NREL_PAPER, 'target_publication_parents': len(targeted),
        'target_publication_assays': sum(len(r['assays']) for r in targeted),
        'target_publication_measurements': sum(len(a['measurements']) for r in targeted for a in r['assays']),
        'target_publication_unique_proteins': len({r['protein_id'] for r in targeted}),
        'nominal_global_temperature_ph_overlap_assays': sum((r['assay']['conditions'].get('temperature_c'),r['assay']['conditions'].get('ph')) in nominal_global for r in selected),
        'name_candidate_assays': len(candidate_assays), 'name_condition_matched_assays': len(matched_assays),
        'name_condition_matched_nrel_cells': len(matched_cells), 'name_condition_pairs': pairs,
        'candidate_assay_name_pairs_without_nominal_match': no_nominal,
        'comparison_categories': dict(classifications), 'comparison_identity_status_counts': dict(identity_counts),
        'missing_field_pair_counts': dict(missing), 'incompatible_field_pair_counts': dict(incompatible),
        'verified_duplicate_cross_source_experiments': documented_duplicates,
    }
    return outcome('SUCCESS', result, [nrel['source'], snapshot['source']], [*LIMITATIONS,
        'Name/alias candidates and shared study DOI are not verified protein or experiment duplicates.',
        'Comparability counts use nominal temperature/pH assay–NREL-cell pairs, not independent experiments or individual measurements.',
        'Raw CSV cells have no authenticated publication-cell, original-experiment or unit mapping; no duplicate is removed or measurement pooled.',
        'A parent live sequence match does not verify the sequence/construct of every curated assay.',
        'All live identity details are transient; only aggregate coverage/outcomes may enter a saved report.'], 'COMPUTED')
