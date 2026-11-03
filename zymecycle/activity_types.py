"""Build 07A audit contracts, separate from measured NREL observations and agent tools."""
from __future__ import annotations

from typing import Literal, TypedDict
from .contracts import JSONValue, Source


class SnapshotCounts(TypedDict):
    parents: int
    assays: int
    representative_assays: int
    measurements: int


class ActivitySnapshot(TypedDict):
    records: list[dict[str, JSONValue]]
    schema_version: str
    sha256: str
    counts: SnapshotCounts
    source: Source


class AssayReference(TypedDict):
    parent_index: int
    assay_index: int
    protein_id: int
    protein_name: str
    publication_doi: str
    assay: dict[str, JSONValue]
    source: Source


class ProteinVerification(TypedDict):
    pazy_protein_id: int
    nrel_identifier: str
    identity: Literal['EXACT_SEQUENCE_MATCH', 'DIFFERENT_SEQUENCE', 'INSUFFICIENT_INFORMATION']
    provenance: list[Source]


class ExperimentDescriptor(TypedDict):
    protein_id: str
    publication_doi: str | None
    experiment_locator: str | None
    fields: dict[str, JSONValue]
    source: Source


class CrossComparability(TypedDict):
    category: Literal['VERIFIED_COMPARABLE', 'LIMITED_COMPARISON', 'NOT_COMPARABLE', 'INSUFFICIENT_INFORMATION']
    missing_fields: list[str]
    incompatible_fields: list[str]
    identity_status: str
    duplicate_status: Literal['SAME_DOCUMENTED_EXPERIMENT', 'SHARED_PUBLICATION_ONLY', 'DISTINCT_PUBLICATIONS', 'UNRESOLVED']
    quantitative_comparison_supported: bool
    causal_claim_supported: Literal[False]
