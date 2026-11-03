"""Transient, whitelisted PAZy metadata and explicit embedded literature links."""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlencode

from .contracts import (JSONValue, LiteratureRecord, LiteratureReference, Organism, Status,
                        PazyProtein, Source, Substrate, ToolResult)
from .pazy import BASE_URL, SOURCE, PazyClient, PazyError
from .results import outcome

LIMITATIONS = [
    'PAZy is the live source; returned records and sequences must never be persisted or logged.',
    'Curated protein/substrate associations are not activity measurements or performance predictions.',
    'Only observed fields and explicit embedded literature relationships are projected; no assay data is retrieved.',
    'Missing metadata is unknown; no relationships are inferred from names or the general literature catalog.',
]
# Retains timing only, not records, cookies, responses or a data cache.
_DEFAULT_CLIENT = PazyClient()


def _source(locator: str) -> Source:
    """Build a PAZy source record from the request URL before any fetch."""
    return {'name': 'PAZy', 'url': 'https://www.pazy.eu/', 'doi': None,
            'locator': locator, 'attribution': SOURCE,
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}


def _text(record: dict, field: str) -> str | None:
    """Read one nullable string field; non-string values mean the live schema changed."""
    value = record.get(field)
    if value is not None and not isinstance(value, str):
        raise PazyError('PAZy metadata schema changed: expected nullable string')
    return value


def _reference(value: JSONValue) -> LiteratureReference:
    """Project one literature link: full record only when the embedded shape verifies.

    Purpose: accepts the EMBEDDED shape (id plus doi/author/title/year)
        and keeps bare IDs or malformed dicts as UNRESOLVED, so partial
        data is never presented as a citable reference."""
    identifier = value if type(value) is int and value > 0 else None
    if isinstance(value, dict):
        raw_id = value.get('id')
        identifier = raw_id if type(raw_id) is int and raw_id > 0 else None
        fields = ('doi', 'author', 'title')
        if (identifier is not None and all(field in value for field in (*fields, 'year'))
                and all(value[field] is None or isinstance(value[field], str) for field in fields)
                and (value['year'] is None or type(value['year']) is int)):
            record: LiteratureRecord = {
                'id': identifier, 'doi': value['doi'], 'author': value['author'],
                'title': value['title'], 'year': value['year'],
            }
            return {'reference_id': identifier, 'resolution': 'EMBEDDED', 'record': record}
    return {'reference_id': identifier, 'resolution': 'UNRESOLVED', 'record': None}


def _protein(record: dict, source: Source) -> PazyProtein:
    """Project the allowlisted protein fields; reject anything else.

    Purpose: converts one raw API record into a PazyProtein with only
        id, name, sequence, UniProt accession, organism, substrates, and
        literature. Unknown fields are dropped (never followed as URLs)
        and mistyped fields raise, so schema drift fails loudly.
    Raises:
        PazyError: when any projected field has an unexpected shape."""
    if type(record.get('id')) is not int or record['id'] < 1:
        raise PazyError('PAZy metadata schema changed: invalid protein identifier')
    literature = record.get('literature')
    if literature is not None and not isinstance(literature, list):
        raise PazyError('PAZy metadata schema changed: expected literature list')
    organism: Organism | None = None
    raw_organism = record.get('organism')
    if raw_organism is not None:
        if not isinstance(raw_organism, dict) or type(raw_organism.get('id')) is not int:
            raise PazyError('PAZy metadata schema changed: invalid organism')
        organism = {'id': raw_organism['id'], 'scientific_name': _text(raw_organism, 'scientific_name'),
                    'domain': _text(raw_organism, 'domain'), 'phylum': _text(raw_organism, 'phylum')}
    substrates: list[Substrate] | None = None
    raw_substrates = record.get('substrates')
    if raw_substrates is not None:
        if not isinstance(raw_substrates, list):
            raise PazyError('PAZy metadata schema changed: invalid substrates')
        substrates = []
        for item in raw_substrates:
            if (not isinstance(item, dict) or type(item.get('id')) is not int or
                    (item.get('oligomer_only') is not None and type(item['oligomer_only']) is not bool)):
                raise PazyError('PAZy metadata schema changed: invalid substrate entry')
            substrates.append({'id': item['id'], 'name': _text(item, 'name'),
                               'abbreviation': _text(item, 'abbreviation'),
                               'oligomer_only': item.get('oligomer_only')})
    return {'id': record['id'], 'name': _text(record, 'name'),
            'amino_acid_sequence': _text(record, 'amino_acid_sequence'),
            'uniprot_accession': _text(record, 'uniprot_accession'),
            'organism': organism, 'substrates': substrates,
            'literature': [_reference(value) for value in literature] if literature is not None else None,
            'source': source}


def search_pazy_pet(limit: int = 3, *, name: str | None = None, client: PazyClient | None = None) -> ToolResult[list[PazyProtein]]:
    """Search live PAZy for PET proteins (one page, at most 3 records).

    Purpose: bounded candidate discovery. The result is transient
        caller-owned memory: IDs feed get_pazy_metadata, but no record may
        be saved, logged, or cached anywhere.
    Args:
        limit: 1..3 records to return.
        name: optional documented API name filter for targeted checks.
        client: injected PazyClient for tests; production uses the shared
            default client (excluded from the JSON tool schema).
    Returns:
        ToolResult[list[PazyProtein]]; each protein carries id, name,
            sequence, organism, substrates, literature, and source.
    Statuses:
        SUCCESS with matches; NO_DATA when the page is empty; ERROR on
            bad limits, transport failure, or schema drift.
    Evidence category: None (metadata retrieval, not measured activity).
    Limitations:
        Not an exhaustive catalog; a name filter match never establishes
        protein identity — only compare_identity does.
    Flow: Receives external limit int, optional candidate name str and PazyClient; returns ToolResult[list[PazyProtein]] for get_pazy_metadata() selection."""
    query = {'substrate': 'PET', 'limit': limit, 'offset': 0}
    if name is not None:
        query['name'] = name
    source = _source(f'{BASE_URL}/proteins/?{urlencode(query)}')
    if type(limit) is not int or not 1 <= limit <= 3:
        return outcome('ERROR', None, [source], LIMITATIONS, None, 'Search limit must be 1 through 3')
    try:
        records = (client or _DEFAULT_CLIENT).search_pet_proteins(limit=limit, max_pages=1, name=name)
        proteins = [_protein(record, source) for record in records]
    except (PazyError, ValueError, KeyError):
        return outcome('ERROR', None, [source], LIMITATIONS, None,
                       'PAZy search failed, input limit was invalid, or the live schema changed')
    limitations = [*LIMITATIONS, 'Search is bounded to one page of at most 3 results, not an exhaustive catalog.']
    return outcome('SUCCESS' if proteins else 'NO_DATA', proteins, [source], limitations, None)


def get_pazy_metadata(protein_id: int, *, client: PazyClient | None = None) -> ToolResult[PazyProtein]:
    """Fetch one live PAZy protein record by ID (transient).

    Purpose: detail lookup for a search-selected ID. The returned record
        (including its sequence field) lives only in caller memory for
        immediate literature or identity projection.
    Args:
        protein_id: positive integer from a same-session search.
        client: injected PazyClient for tests (excluded from JSON schema).
    Returns:
        ToolResult[PazyProtein] with the projected protein record.
    Statuses:
        SUCCESS normally; ERROR on transport failure, bad ID, or schema drift.
    Evidence category: None (metadata retrieval).
    Flow: Receives selected protein_id int from search_pazy_pet() or external input; returns ToolResult[PazyProtein] for get_associated_literature() or compare_identity()."""
    source = _source(f'{BASE_URL}/proteins/{protein_id}/')
    try:
        record = (client or _DEFAULT_CLIENT).get_protein(protein_id)
        protein = _protein(record, source)
    except (PazyError, ValueError):
        return outcome('ERROR', None, [source], LIMITATIONS, None,
                       'PAZy metadata lookup failed, protein ID was invalid, or the live schema changed')
    return outcome('SUCCESS', protein, [source], LIMITATIONS, None)


def _literature_result(status: Status, references: list[LiteratureReference] | None,
                       source: Source, limitations: list[str], total: int | None,
                       message: str | None = None) -> ToolResult[list[LiteratureReference]]:
    """Attach reference_counts (total/returned/omitted/unresolved) to a literature result."""
    result = outcome(status, references, [source], limitations, None, message)
    returned = len(references or [])
    result['reference_counts'] = {
        'total': total, 'returned': returned, 'omitted': total - returned if total is not None else None,
        'unresolved_returned': sum(item['resolution'] == 'UNRESOLVED' for item in references or []),
    }
    return result


def get_associated_literature(protein: PazyProtein, limit: int = 5) -> ToolResult[list[LiteratureReference]]:
    """Return a protein's explicit embedded literature links (at most 5).

    Purpose: exposes only the literature relationships the record itself
        embeds. Missing relationships stay unknown and malformed ones stay
        UNRESOLVED; the general catalog is never searched to assign papers.
    Args:
        protein: transient PazyProtein from get_pazy_metadata().
        limit: 1..5 references to return (first N in record order).
    Returns:
        ToolResult[list[LiteratureReference]] plus a 'reference_counts'
            envelope key (total/returned/omitted/unresolved_returned) so
            truncation is always visible.
    Statuses:
        SUCCESS when all returned links resolved; PARTIAL when some are
            unresolved, truncated, or the relationship is absent; NO_DATA
            when the record embeds an empty list; ERROR on a bad limit.
    Flow: Receives transient PazyProtein from get_pazy_metadata(); returns ToolResult[list[LiteratureReference]] as final explicit references with omitted-reference counts."""
    if type(limit) is not int or not 1 <= limit <= 5:
        return _literature_result('ERROR', None, protein['source'], LIMITATIONS, None, 'Reference limit must be 1 through 5')
    references = protein['literature']
    if references is None:
        return _literature_result('PARTIAL', [], protein['source'], LIMITATIONS, None,
                                  'Literature relationship absent; reference availability is unknown')
    limitations = LIMITATIONS.copy()
    if len(references) > limit:
        limitations.append(f'Only the first {limit} of {len(references)} explicit associations are returned.')
    selected = references[:limit]
    unresolved = any(item['resolution'] == 'UNRESOLVED' for item in selected)
    status = 'PARTIAL' if unresolved or len(references) > limit else 'SUCCESS' if selected else 'NO_DATA'
    return _literature_result(status, selected, protein['source'], limitations, len(references),
                              'Unsupported identifier-only or malformed references remain unresolved' if unresolved else None)
