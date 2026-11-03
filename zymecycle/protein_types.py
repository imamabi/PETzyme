"""Local sequence-representation contracts; none implies measured activity."""
from __future__ import annotations

from typing import Literal, Protocol, TypedDict
from .contracts import Source


class PreparedSequence(TypedDict):
    identifier: str | None
    normalized_sequence: str | None
    sequence_sha256: str | None
    sequence_length: int
    validation: Literal['VALID', 'MISSING', 'AMBIGUOUS_RESIDUES', 'UNSUPPORTED_RESIDUES', 'TOO_LONG']
    flagged_symbols: list[str]
    source_row: int | None
    csv_sha256: str | None
    provenance: list[Source]


class ModelSpec(TypedDict):
    checkpoint: str
    revision: str
    license: str
    dimension: int
    layer: int
    pooling: str
    dtype: str


class ProteinEmbedding(TypedDict):
    identifier: str | None
    sequence_sha256: str
    sequence_length: int
    vector: list[float]
    model: ModelSpec
    device: str
    fallback: str | None
    source_row: int | None
    csv_sha256: str | None
    provenance: list[Source]


class EmbeddingRuntime(Protocol):
    spec: ModelSpec
    device: str
    fallback: str | None

    def encode(self, sequence: str) -> list[float]:
        """Flow: Receives validated str from embed_protein(); returns local vector list[float] for embed_protein()."""
        ...


class EmbeddingComparison(TypedDict):
    cosine_similarity: float
    model: ModelSpec


class SimilarProtein(TypedDict):
    identifier: str
    source_row: int
    cosine_similarity: float
    provenance: list[Source]


class CandidateExclusion(TypedDict):
    identifier: str
    source_row: int
    reason: str


class SearchCoverage(TypedDict):
    total_records: int
    query_excluded_records: int
    candidate_pool_records: int
    candidate_window_records: int
    outside_window_records: int
    eligible_candidates: int
    searched_candidates: int
    excluded_in_window: int
    returned_neighbors: int
    selection: str
    scope: str


class SimilaritySearch(TypedDict):
    query_identifier: str
    query_source_row: int
    csv_sha256: str
    model: ModelSpec
    device: str
    fallback: str | None
    devices_used: list[str]
    fallbacks: list[str]
    coverage: SearchCoverage
    exclusions: list[CandidateExclusion]
    neighbors: list[SimilarProtein]
