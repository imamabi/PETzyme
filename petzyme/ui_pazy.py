"""Transient PAZy display projections; no persistence, inference or cross-source merging."""
from __future__ import annotations

import re
from typing import TypedDict
from urllib.parse import quote

from .contracts import ReferenceCounts, ToolResult
from .metadata import get_associated_literature, search_pazy_pet
from .pazy import PazyClient
from .results import outcome

PAZY_URL = 'https://www.pazy.eu/'
ARTICLE_DOI = '10.1093/nar/gkag1026'


class PublicationView(TypedDict):
    title: str | None
    author: str | None
    year: int | None
    doi_url: str | None


class ProteinView(TypedDict):
    name: str | None
    organism: str | None
    domain: str | None
    phylum: str | None
    pet_associations: list[str]
    publications: list[PublicationView]
    reference_counts: ReferenceCounts
    unavailable: list[str]


def doi_link(doi: str | None) -> str | None:
    """Flow: Receives embedded DOI str | None from discover_pazy_pet(); returns HTTPS str | None for render_pazy_discovery()."""
    if not doi:
        return None
    doi = re.sub(r'^https://doi\.org/', '', doi.strip(), flags=re.IGNORECASE)
    if not re.fullmatch(r'10\.\d{4,9}/\S+', doi):
        return None
    return 'https://doi.org/' + quote(doi, safe='/')


def discover_pazy_pet(limit: int = 3, *, client: PazyClient | None = None) -> ToolResult[list[ProteinView]]:
    """Discover live PAZy PET proteins as transient display views (search + literature).

    Purpose: composes search_pazy_pet with per-protein literature for
        the discovery tab. Views are transient caller memory like all
        live PAZy data — rendered, never stored.
    Flow: Receives external limit int and optional PazyClient; returns transient ToolResult[list[ProteinView]] from search_pazy_pet() and get_associated_literature() for render_pazy_discovery()."""
    search = search_pazy_pet(limit, client=client)
    if search['result'] is None:
        return outcome(search['status'], None, search['provenance'], search['limitations'], None, search['message'])
    cards: list[ProteinView] = []
    partial = False
    for protein in search['result']:
        literature = get_associated_literature(protein, limit=3)
        organism = protein['organism'] or {}
        unavailable = []
        for label, value in [('Protein name', protein['name']), ('Source organism', organism.get('scientific_name')),
                             ('Domain', organism.get('domain')), ('Phylum', organism.get('phylum'))]:
            if not value:
                unavailable.append(label + ' unavailable.')
        associations = []
        for substrate in protein['substrates'] or []:
            if substrate['abbreviation'] == 'PET':
                qualifier = ('oligomer-only annotation' if substrate['oligomer_only'] is True else
                             'not marked oligomer-only; this does not rule out oligomer activity' if substrate['oligomer_only'] is False else
                             'oligomer-only status unavailable')
                associations.append((substrate['name'] or 'PET') + ': ' + qualifier + '.')
        if not associations:
            unavailable.append('PET filter matched, but an explicit PET substrate annotation was unavailable; no association inferred.')
        publications: list[PublicationView] = []
        for reference in literature['result'] or []:
            record = reference['record']
            if record is not None:
                publications.append({'title': record['title'], 'author': record['author'], 'year': record['year'],
                                     'doi_url': doi_link(record['doi'])})
                if any(record[f] is None for f in ('title', 'author', 'year')):
                    unavailable.append('Some associated publication metadata is unavailable.')
                if doi_link(record['doi']) is None:
                    unavailable.append('An associated publication has no usable DOI link.')
        counts = literature['reference_counts']
        if counts['total'] is None:
            unavailable.append('Literature relationship unavailable; total and omitted reference counts are unknown.')
        elif counts['total'] == 0:
            unavailable.append('No literature associations supplied for this protein; this does not prove no publications exist.')
        if counts['unresolved_returned']:
            unavailable.append('Some explicitly associated references could not be resolved; no catalog entries substituted.')
        partial |= bool(unavailable or counts['omitted'])
        cards.append({'name': protein['name'], 'organism': organism.get('scientific_name'),
                      'domain': organism.get('domain'), 'phylum': organism.get('phylum'),
                      'pet_associations': associations, 'publications': publications,
                      'reference_counts': counts, 'unavailable': list(dict.fromkeys(unavailable))})
    return outcome('PARTIAL' if partial else search['status'], cards, search['provenance'], search['limitations'], None)
