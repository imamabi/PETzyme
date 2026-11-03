"""Manual audit entry point; emits aggregate diagnostics, never live protein/API records."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from petzyme.activity_audit import audit_pazy_pet_assays
from petzyme.activity_snapshot import load_pazy_activity_snapshot
from petzyme.cross_overlap import candidate_protein_pairs, identify_cross_source_overlap, verify_snapshot_identities
from petzyme.observations import load_nrel


def main() -> None:
    """Flow: Receives external --verify-live CLI authorization; composes load_pazy_activity_snapshot(), audit_pazy_pet_assays(), verify_snapshot_identities() and identify_cross_source_overlap() into final aggregate JSON stdout only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-live', action='store_true', help='Perform at most three targeted PAZy protein lookups; never paid inference')
    args = parser.parse_args()
    snapshot = load_pazy_activity_snapshot()
    nrel = load_nrel()
    if snapshot['result'] is None or nrel['result'] is None:
        print(json.dumps({'status': 'ERROR', 'message': 'Verified local source unavailable; nothing downloaded or substituted.'}))
        return
    snapshot, nrel = snapshot['result'], nrel['result']
    inventory = audit_pazy_pet_assays(snapshot)['result']
    proofs = []
    http = {'requests': 0, 'statuses': Counter()}
    verification_status = 'NOT_REQUESTED'
    if args.verify_live:
        import httpx
        from petzyme.pazy import PazyClient
        class CountTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                """Flow: Receives httpx.Request during PazyClient._get(); returns HTTPTransport.handle_request() httpx.Response for that lookup, retaining only numeric counts."""
                self.transport = httpx.HTTPTransport(retries=0)
                http['requests'] += 1
                response = self.transport.handle_request(request)
                http['statuses'][str(response.status_code)] += 1
                return response
            def close(self) -> None:
                """Flow: Receives httpx.Client cleanup during PazyClient._get(); returns None after closing the underlying HTTPTransport."""
                self.transport.close()
        available = candidate_protein_pairs(snapshot, nrel)
        candidates = []
        for name in ('ESM050', 'DP043', 'DP003'):
            matching = [p for p in available if p[1] == name]
            if len(matching) == 1: candidates.append(matching[0])
        result = verify_snapshot_identities(snapshot, nrel, candidates, client=PazyClient(transport=CountTransport()))
        verification_status, proofs = result['status'], result['result'] or []
    assessed = identify_cross_source_overlap(snapshot, nrel, proofs)
    if assessed['result'] is None:
        print(json.dumps({'status': 'ERROR', 'message': 'Cross-source audit input failed validation; no live records retained.'}))
        return
    overlap = assessed['result']
    # Never print the candidate-level list or proof provenance/IDs, even on a live run.
    overlap = {k: v for k, v in overlap.items() if k not in ('candidate_pairs', 'persistence')}
    for key in ('analyte_inventory', 'method_inventory'):
        inventory[key + '_distinct'] = len(inventory[key])
        inventory[key] = dict(Counter(inventory[key]).most_common(12))
    print(json.dumps({'status': 'SUCCESS', 'inventory': inventory, 'overlap': overlap,
                      'live_verification_status': verification_status, 'live_http': http}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
