"""Build 01 demonstration: local observations and optional schema-only live checks."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from petzyme.contracts import JSONValue, ToolResult
from petzyme.metadata import get_associated_literature, get_pazy_metadata, search_pazy_pet
from petzyme.observations import load_nrel, parse_nrel, summarize_condition


def nrel_examples(csv_path: str) -> dict[str, ToolResult]:
    """Flow: Receives external NREL CSV path str; returns dict[str, ToolResult] of two exact-condition summaries for main()."""
    loaded = load_nrel(csv_path)
    if loaded['result'] is None:
        return {'local_error': loaded}
    parsed = parse_nrel(loaded['result'])
    if parsed['result'] is None:
        return {'local_error': parsed}
    observations = parsed['result']['observations']
    return {'scenario_a': summarize_condition(observations, 40, 6.5, 'CRYSTALLINE_POWDER'),
            'scenario_b': summarize_condition(observations, 50, 8, 'CRYSTALLINE_POWDER')}


def pazy_smoke_summary() -> dict[str, JSONValue]:
    """Flow: Receives external live-check invocation; returns dict[str, JSONValue] of status and schema only for main()."""
    summary: dict[str, JSONValue] = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                                     'source': 'PAZy, https://www.pazy.eu/'}
    search = search_pazy_pet(limit=1)
    summary['search_status'] = search['status']
    if not search['result']:
        summary['message'] = search['message'] or 'No live PET protein returned'
        return summary
    metadata = get_pazy_metadata(search['result'][0]['id'])
    summary['metadata_status'] = metadata['status']
    if metadata['result'] is None:
        summary['message'] = metadata['message']
        return summary
    protein = metadata['result']
    summary['metadata_fields'] = sorted(protein)
    references = get_associated_literature(protein, limit=5)
    summary['reference_status'] = references['status']
    summary['reference_counts'] = references['reference_counts']
    summary['explicit_association_count'] = len(protein['literature']) if protein['literature'] is not None else None
    summary['returned_reference_count'] = len(references['result'] or [])
    summary['embedded_reference_count'] = sum(item['resolution'] == 'EMBEDDED' for item in references['result'] or [])
    summary['unresolved_reference_count'] = sum(item['resolution'] == 'UNRESOLVED' for item in references['result'] or [])
    summary['reference_limitations'] = references['limitations']
    # No record identifier, name, title, DOI, sequence, hash or response body escapes this summary.
    return summary


def main(argv: list[str] | None = None) -> int:
    """Flow: Receives external CLI arguments list[str] | None; returns exit status int after printing NREL summaries and optional PAZy schema/status."""
    parser = argparse.ArgumentParser(description='ZymeCycle Build 01 demonstrations')
    parser.add_argument('--csv', default='data/nrel/label_data.csv')
    parser.add_argument('--live-pazy', action='store_true')
    args = parser.parse_args(argv)
    examples = nrel_examples(args.csv)
    print(json.dumps({'nrel': examples}, indent=2, allow_nan=False))
    if args.live_pazy:
        print(json.dumps({'pazy_smoke': pazy_smoke_summary()}, indent=2, allow_nan=False))
    return 1 if 'local_error' in examples else 0


if __name__ == '__main__':
    raise SystemExit(main())
