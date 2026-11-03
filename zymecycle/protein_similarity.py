"""Bounded NREL representation search, independent of language-model providers."""
from __future__ import annotations

from .contracts import NrelTable, ToolResult
from .esm import LIMITATIONS, compare_protein_embeddings, embed_protein
from .observations import select_nrel_protein
from .protein_sequences import prepare_protein_sequence
from .protein_types import CandidateExclusion, SimilarProtein, SimilaritySearch
from .results import outcome

SEARCH_LIMITATIONS = [*LIMITATIONS,
    'Candidates are the first bounded nonquery records in original CSV order; this is not a whole-dataset neighbor ranking.',
    'Eligible/searched/excluded counts refer to that candidate window; unexamined records may be more similar.',
    'Similarity scores are not sequence-identity percentages; no activity or condition comparison is performed.',
]


def find_similar_nrel_proteins(table: NrelTable, enzyme_identifier: str,
                               top_k: int = 5, max_candidates: int = 8) -> ToolResult[SimilaritySearch]:
    """Rank the most similar NREL proteins to one query within a bounded window.

    Purpose: ESM-2 similarity search over the first `max_candidates`
        non-query CSV records in original order — explicitly not a
        whole-dataset ranking. Ineligible candidates (bad identifier,
        invalid sequence, inference failure) are listed in 'exclusions'
        with reasons instead of being silently dropped.
    Args:
        table: NrelTable from load_nrel().
        enzyme_identifier: exact query label (must resolve uniquely).
        top_k: 1..10 neighbors to return.
        max_candidates: 1..32 window size in original CSV order.
    Returns:
        ToolResult[SimilaritySearch] with query identity, model/device
            info, a 'coverage' breakdown (pool/window/eligible/searched/
            excluded counts), 'exclusions', and 'neighbors' sorted by
            descending cosine (ties by identifier, then row).
    Statuses:
        SUCCESS with neighbors; PARTIAL when eligible candidates failed;
            NO_DATA when no neighbor resulted; ERROR on bad bounds or an
            unresolvable/unembeddable query.
    Example:
        >>> out = find_similar_nrel_proteins(load_nrel()['result'], 'LCC-ICCG',
        ...     top_k=2, max_candidates=4)
        >>> out['result']['coverage']['scope']
        'BOUNDED_CANDIDATE_WINDOW'
    Limitations:
        Scores are not identity percentages; unexamined records outside
            the window may be more similar; no activity comparison.
    Flow: Receives NrelTable from load_nrel() and external exact identifier/bounds; returns ToolResult[SimilaritySearch] for project_evidence(), recommend_next_experiments() or a final local ranking."""
    sources = [table['source']]
    if (type(top_k) is not int or not 1 <= top_k <= 10 or
            type(max_candidates) is not int or not 1 <= max_candidates <= 32):
        return outcome('ERROR', None, sources, SEARCH_LIMITATIONS, 'COMPUTED', 'Bounds: top_k 1–10; max_candidates 1–32')
    query = select_nrel_protein(table, enzyme_identifier)
    if query['result'] is None:
        return outcome(query['status'], None, sources, SEARCH_LIMITATIONS, 'COMPUTED', query['message'])
    prepared = prepare_protein_sequence(query['result'])
    if prepared['status'] != 'SUCCESS':
        return outcome(prepared['status'], None, sources, SEARCH_LIMITATIONS, 'COMPUTED', prepared['message'])
    embedded_query = embed_protein(prepared['result'])
    if embedded_query['status'] != 'SUCCESS':
        return outcome('ERROR', None, embedded_query['provenance'], SEARCH_LIMITATIONS, 'COMPUTED', embedded_query['message'])
    query_embedding = embedded_query['result']
    devices = {query_embedding['device']}
    fallbacks = [query_embedding['fallback']] if query_embedding['fallback'] else []
    sources = embedded_query['provenance']
    query_row = query['result']['source_row']
    identifier_index = table['columns'].index('')
    # Source-order window, not a ranking pre-filter: the first N non-query rows, whatever they are.
    pool = [(i,row[identifier_index]) for i,row in enumerate(table['rows'],start=2) if i != query_row]
    window = pool[:max_candidates]
    neighbors: list[SimilarProtein] = []
    exclusions: list[CandidateExclusion] = []
    eligible, failed = 0, 0
    for source_row, identifier in window:
        reason = None
        candidate = select_nrel_protein(table, identifier)
        if candidate['status'] != 'SUCCESS':
            reason = 'AMBIGUOUS_OR_MISSING_IDENTIFIER'
        else:
            candidate_sequence = prepare_protein_sequence(candidate['result'])
            if candidate_sequence['status'] != 'SUCCESS':
                reason = candidate_sequence['result']['validation']
            else:
                eligible += 1
                embedding = embed_protein(candidate_sequence['result'])
                if embedding['status'] != 'SUCCESS':
                    reason, failed = 'INFERENCE_FAILED', failed + 1
                else:
                    devices.add(embedding['result']['device'])
                    if embedding['result']['fallback'] and embedding['result']['fallback'] not in fallbacks:
                        fallbacks.append(embedding['result']['fallback'])
                    comparison = compare_protein_embeddings(query_embedding, embedding['result'])
                    if comparison['status'] != 'SUCCESS':
                        reason, failed = 'COMPARISON_FAILED', failed + 1
                    else:
                        neighbors.append({'identifier': identifier, 'source_row': source_row,
                                          'cosine_similarity': comparison['result']['cosine_similarity'],
                                          'provenance': embedding['provenance']})
        if reason:
            exclusions.append({'identifier': identifier, 'source_row': source_row, 'reason': reason})
    searched = len(neighbors)
    neighbors.sort(key=lambda item: (-item['cosine_similarity'], item['identifier'], item['source_row']))
    result: SimilaritySearch = {
        'query_identifier': enzyme_identifier, 'query_source_row': query_row,
        'csv_sha256': table['csv_sha256'],
        'model': query_embedding['model'], 'device': query_embedding['device'], 'fallback': query_embedding['fallback'],
        'devices_used': sorted(devices), 'fallbacks': fallbacks,
        'coverage': {'total_records': len(table['rows']), 'query_excluded_records': 1,
                     'candidate_pool_records': len(pool), 'candidate_window_records': len(window),
                     'outside_window_records': len(pool) - len(window), 'eligible_candidates': eligible,
                     'searched_candidates': searched, 'excluded_in_window': len(exclusions),
                     'returned_neighbors': min(top_k,searched),
                     'selection': 'First nonquery CSV records in original order; ties by identifier then source row',
                     'scope': 'BOUNDED_CANDIDATE_WINDOW'},
        'exclusions': exclusions, 'neighbors': neighbors[:top_k],
    }
    return outcome('PARTIAL' if failed else 'SUCCESS' if neighbors else 'NO_DATA', result,
                   sources, SEARCH_LIMITATIONS, 'COMPUTED',
                   'One or more eligible candidates failed inference/comparison' if failed else None)
