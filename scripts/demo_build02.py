"""Build 02 demonstrations; PAZy records and identity objects remain transient."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from zymecycle.comparability import check_comparability
from zymecycle.conditions import investigate_conditions
from zymecycle.contracts import JSONValue, NrelTable, ToolResult
from zymecycle.evidence import assess_evidence
from zymecycle.gaps import identify_evidence_gaps
from zymecycle.identity import compare_identity, sequence_fingerprint
from zymecycle.metadata import get_associated_literature, get_pazy_metadata, search_pazy_pet
from zymecycle.observations import filter_observations, load_nrel, parse_nrel, select_nrel_protein
from zymecycle.pazy import PazyClient
from zymecycle.results import outcome

A = {'temperature_c': 40, 'ph': 6.5, 'pet_form': 'CRYSTALLINE_POWDER'}
B = {**A, 'temperature_c': 60}


def local_demonstrations(table: NrelTable) -> dict[str, ToolResult]:
    """Flow: Receives NrelTable from load_nrel(); returns dict[str, ToolResult] of deterministic scientific demonstrations for main()."""
    parsed = parse_nrel(table)
    if parsed['result'] is None:
        return {'local_error': parsed}
    observations = parsed['result']['observations']
    exact = filter_observations(observations, **A)['result'] or []
    selected = [o for o in exact if o['enzyme_identifier'] in ('LCC-ICCG', 'ESM050')]
    film = {**A, 'pet_form': 'AMORPHOUS_FILM'}
    missing = {**A, 'temperature_c': 50, 'ph': 8}
    investigation = investigate_conditions(observations, A, B, enzyme_identifier='ESM050')
    return {
        'within_condition': check_comparability(selected),
        'different_pet_forms': investigate_conditions(observations, A, film, enzyme_identifier='ESM050'),
        'missing_condition': investigate_conditions(observations, A, missing, enzyme_identifier='ESM050'),
        'limitations': assess_evidence(filter_observations(observations, enzyme_identifier='LCC-ICCG')['result'] or [],
                                       expected_conditions=[A, B, missing]),
        'temperature_investigation': investigation,
        'temperature_evidence_gaps': identify_evidence_gaps(investigation['result']),
    }


def bounded_identity_check(table: NrelTable, *, name: str | None = None, client: PazyClient | None = None) -> ToolResult[dict[str, JSONValue]]:
    """Flow: Receives NrelTable from load_nrel() and optional candidate name str/PazyClient; returns transient ToolResult[dict[str, JSONValue]] for privacy_summary()."""
    by_hash: dict[str, str] = {}
    columns = table['columns']
    for row in table['rows']:
        fingerprint = sequence_fingerprint(row[columns.index('sequence')])
        if fingerprint['sha256'] is not None:
            by_hash.setdefault(fingerprint['sha256'], row[columns.index('')])
    search = search_pazy_pet(limit=3, name=name, client=client)
    if search['result'] is None:
        return outcome('ERROR', None, [table['source'], *search['provenance']], search['limitations'], 'COMPUTED', search['message'])
    checked = 0
    for candidate in search['result']:
        checked += 1
        fingerprint = sequence_fingerprint(candidate['amino_acid_sequence'])
        identifier = by_hash.get(fingerprint['sha256'])
        if identifier is None:
            continue
        selected = select_nrel_protein(table, identifier)
        detail = get_pazy_metadata(candidate['id'], client=client)
        if selected['result'] is None or detail['result'] is None:
            return outcome('ERROR', None, [table['source'], *detail['provenance']], detail['limitations'], 'COMPUTED',
                           'Selected live protein detail could not be validated')
        identity = compare_identity(selected['result'], detail['result'])
        if identity['result']['identity'] != 'EXACT_SEQUENCE_MATCH':
            continue
        references = get_associated_literature(detail['result'])
        return outcome('SUCCESS', {'identity': identity['result'], 'candidates_checked': checked,
                                  'reference_status': references['status'],
                                  'reference_counts': references['reference_counts']},
                       identity['provenance'], identity['limitations'], 'COMPUTED')
    return outcome('NO_DATA', {'identity': None, 'candidates_checked': checked}, [table['source'], *search['provenance']],
                   search['limitations'], 'COMPUTED', 'No exact shared sequence found in the bounded three-candidate window')


def privacy_summary(check: ToolResult[dict[str, JSONValue]]) -> dict[str, JSONValue]:
    """Flow: Receives transient ToolResult[dict[str, JSONValue]] from bounded_identity_check(); returns dict[str, JSONValue] of aggregate status only for main()."""
    result = check['result'] or {}
    identity = result.get('identity')
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': check['status'],
            'identity_outcome': identity['identity'] if identity else None,
            'candidates_checked': result.get('candidates_checked'),
            'both_source_identities_preserved_in_memory': bool(identity),
            'reference_status': result.get('reference_status'), 'reference_counts': result.get('reference_counts'),
            'message': check['message']}


def main(argv: list[str] | None = None) -> int:
    """Flow: Receives external CLI arguments list[str] | None; returns exit status int after local evidence output and optional aggregate-only live identity status."""
    parser = argparse.ArgumentParser(description='ZymeCycle Build 02 evidence demonstrations')
    parser.add_argument('--csv', default='data/nrel/label_data.csv')
    parser.add_argument('--live-pazy', action='store_true')
    parser.add_argument('--pazy-candidate-name', default=None)
    args = parser.parse_args(argv)
    loaded = load_nrel(args.csv)
    if loaded['result'] is None:
        print(json.dumps({'local_error': loaded}))
        return 1
    print(json.dumps({'local': local_demonstrations(loaded['result'])}, indent=2, allow_nan=False))
    if args.live_pazy:
        print(json.dumps({'live_identity': privacy_summary(bounded_identity_check(loaded['result'], name=args.pazy_candidate_name))}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
