"""Local identity triage; name candidates never authenticate constructs or function."""

from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from .contracts import NrelTable, Source, ToolResult
from .observations import load_nrel
from .pet_retrieval import ROOT, load_activity_source
from .protein_sequences import prepare_protein_sequence
from .results import outcome

MIN_FRAGMENT_RESIDUES = 20  # Operational search floor, not a biological significance threshold.
MAX_INPUT_CHARS = 16384
LIMITATIONS = [
    "Names, documented aliases and punctuation-normalized suggestions are candidates, never sequence-identity proofs.",
    "Sequence authentication means equality to a checksum-bound NREL entry; source completeness and experimental construct identity remain unverified.",
    "Sequence identity or a shared fragment proves neither PET activity, function nor experimental equivalence.",
    "Only the existing 20-residue alphabet and 1,022-residue operational bound are supported; ambiguous residues are not substituted.",
    "Fragment candidates require an exact contiguous substring of at least 20 residues; this search floor is not statistical significance or an activity threshold.",
    "No alignment, ESM-2, prediction, protein API, download, cache, logging or persistent sequence storage is used.",
    "PAZy names/aliases come only from the pinned activity archive; it supplies no authenticated local protein sequences. Results are transient.",
]


class IdentityCandidate(TypedDict):
    source_name: str
    identifier: str
    name: str
    basis: list[str]
    source: Source


class IdentityTriage(TypedDict):
    identity_status: Literal["IDENTIFIED_SEQUENCE", "CANDIDATE_MATCH", "UNRESOLVED"]
    sequence_validation: str
    input_format: Literal["RAW", "FASTA"] | None
    sequence_length: int | None
    flagged_symbols: list[str]
    candidates: list[IdentityCandidate]
    total_candidates: int
    omitted_candidates: int
    multiple_source_records: bool
    name_sequence_agreement: Literal["MATCH", "UNVERIFIED", "NOT_APPLICABLE"]
    issues: list[str]
    nrel_records_checked: int
    nrel_sequences_eligible: int
    alias_source_available: bool
    alias_source_checked: bool
    alignment_performed: Literal[False]
    persistence: Literal["TRANSIENT"]


def _parse_input(value: str) -> tuple[str | None, str, str, int, list[str]]:
    """Validate a single FASTA/plain sequence without retaining or interpreting its header."""
    if len(value) > MAX_INPUT_CHARS:
        return None, "INPUT_TOO_LARGE", "RAW", 0, []
    text = value.strip()
    form = "FASTA" if text.startswith(">") else "RAW"
    if form == "FASTA":
        lines = text.splitlines()
        if len(lines) < 2 or not lines[0][1:].strip() or any(">" in line for line in lines[1:]):
            return None, "INVALID_FASTA", form, 0, []
        text = "\n".join(lines[1:])
    prepared = prepare_protein_sequence(text)["result"]
    return (
        prepared["normalized_sequence"],
        prepared["validation"],
        form,
        prepared["sequence_length"],
        prepared["flagged_symbols"],
    )


def load_authenticated_nrel_reference() -> NrelTable | None:
    """Fail closed if the local CSV differs from its recorded source-audit checksum.

    Flow: Receives the pinned local source manifest from triage_protein() or retrieved_observations(); returns authenticated NrelTable | None for those callers without caching.
    """
    try:
        manifest = json.loads((ROOT / "NREL_SOURCE_MANIFEST.json").read_text())
        loaded = load_nrel(str(ROOT / "data/nrel/label_data.csv"))["result"]
        if (
            loaded is None
            or manifest["dataset_doi"] != loaded["source"]["doi"]
            or manifest["extracted_file"] != "data/nrel/label_data.csv"
            or loaded["csv_sha256"] != manifest["unit_reconciliation"]["csv_sha256"]
        ):
            return None
        return loaded
    except (OSError, ValueError, TypeError, KeyError):
        return None


def triage_protein(
    name: str | None = None, sequence_input: str | None = None, limit: int = 3
) -> ToolResult[IdentityTriage]:
    """Triage a name and/or sequence against local NREL rows plus snapshot aliases.

    Purpose: local identity check without any LLM, alignment, ESM-2, or
        network call. Scans every NREL row for exact/normalized name
        matches and exact-sequence/exact-fragment (>=20 residues)
        matches, then scans the pinned activity snapshot's documented
        protein/alias names only. Refuses to run unless the local CSV
        matches the manifest checksum, so triage can never silently use
        a different dataset version.
    Args:
        name: optional ASCII label, at most 120 chars. At least one of
            name/sequence_input is required.
        sequence_input: optional raw or FASTA sequence string (FASTA
            headers are ignored, never interpreted as names).
        limit: 1..5 candidates to return.
    Returns:
        ToolResult[IdentityTriage] with 'identity_status'
            (IDENTIFIED_SEQUENCE requires an exact sequence hit, unique
            or reconciled with the name; else CANDIDATE_MATCH or
            UNRESOLVED), 'sequence_validation', 'candidates' (basis
            lists such as EXACT_NAME, EXACT_SEQUENCE, EXACT_FRAGMENT,
            DOCUMENTED_ALIAS), counts, 'name_sequence_agreement',
            'issues', and 'alignment_performed' (always False).
    Statuses:
        SUCCESS with candidates; NO_DATA when nothing matches; PARTIAL
            when the alias source was unavailable; ERROR on invalid
            input, failed sequence validation, or unverified NREL source
            (the result payload still explains why).
    Example:
        >>> out = triage_protein(name='ESM050')
        >>> (out['status'], out['result']['identity_status'])
        ('SUCCESS', 'CANDIDATE_MATCH')
    Limitations:
        Name matches are candidates, never identity proofs; even an
            exact sequence match does not authenticate the assayed
            construct, function, or activity.
    Flow: Receives external name str | None, raw/FASTA sequence_input str | None and limit int via invoke_tool(); returns transient ToolResult[IdentityTriage] for final local identity review without LLM invocation."""
    value: IdentityTriage = {
        "identity_status": "UNRESOLVED",
        "sequence_validation": "NOT_SUPPLIED",
        "input_format": None,
        "sequence_length": None,
        "flagged_symbols": [],
        "candidates": [],
        "total_candidates": 0,
        "omitted_candidates": 0,
        "multiple_source_records": False,
        "name_sequence_agreement": "NOT_APPLICABLE",
        "issues": [],
        "nrel_records_checked": 0,
        "nrel_sequences_eligible": 0,
        "alias_source_available": False,
        "alias_source_checked": False,
        "alignment_performed": False,
        "persistence": "TRANSIENT",
    }
    if (
        type(limit) is not int
        or not 1 <= limit <= 5
        or name is not None
        and (not isinstance(name, str) or not name.strip() or len(name) > 120 or not name.isascii())
        or sequence_input is not None
        and not isinstance(sequence_input, str)
        or name is None
        and sequence_input is None
    ):
        value["issues"] = ["INVALID_INPUT"]
        return outcome(
            "ERROR",
            value,
            [],
            LIMITATIONS,
            "COMPUTED",
            "Supply a name and/or one raw/FASTA sequence; limit 1–5.",
        )
    sequence = None
    if sequence_input is not None:
        sequence, validation, form, length, symbols = _parse_input(sequence_input)
        value.update(
            sequence_validation=validation,
            input_format=form,
            sequence_length=length,
            flagged_symbols=symbols,
        )
        if validation != "VALID":
            value["issues"] = [validation]
            return outcome(
                "ERROR",
                value,
                [],
                LIMITATIONS,
                "COMPUTED",
                "Sequence input failed validation; no identity search performed.",
            )
        if length < MIN_FRAGMENT_RESIDUES:
            value["issues"].append("SHORT_SEQUENCE_FRAGMENT_SEARCH_DISABLED")
    table = load_authenticated_nrel_reference()
    if table is None:
        value["issues"].append("NREL_SOURCE_UNVERIFIED")
        return outcome(
            "ERROR",
            value,
            [],
            LIMITATIONS,
            "COMPUTED",
            "Pinned NREL reference unavailable or mismatched; no identity proof or fallback.",
        )
    nrel_source = {**table["source"], "locator": table["source"]["url"] + "#file=label_data.csv"}
    sources, candidates, exact_rows, named_rows = [nrel_source], [], set(), set()
    query = name.strip() if name is not None else None

    def normalized(text: str) -> str:
        return re.sub("[^a-z0-9]", "", text.casefold())

    def name_basis(label: str, alias: bool = False) -> str | None:
        if query is None:
            return None
        if query == label:
            return "DOCUMENTED_ALIAS" if alias else "EXACT_NAME"
        if normalized(query) and normalized(query) == normalized(label):
            return "NORMALIZED_NAME_CANDIDATE"
        return None

    id_index, seq_index = table["columns"].index(""), table["columns"].index("sequence")
    value["nrel_records_checked"] = len(table["rows"])
    for row_number, row in enumerate(table["rows"], 2):
        label = row[id_index]
        basis = []
        nominal = name_basis(label)
        if nominal:
            basis.append(nominal)
            if nominal == "EXACT_NAME":
                named_rows.add(row_number)
        reference = prepare_protein_sequence(row[seq_index])["result"]["normalized_sequence"]
        value["nrel_sequences_eligible"] += int(reference is not None)
        if sequence is not None and reference is not None:
            if sequence == reference:
                basis.append("EXACT_SEQUENCE")
                exact_rows.add(row_number)
            elif len(sequence) >= MIN_FRAGMENT_RESIDUES and sequence in reference:
                basis.append("EXACT_FRAGMENT")
        if basis:
            candidates.append(
                {
                    "source_name": "NREL",
                    "identifier": label,
                    "name": label,
                    "basis": basis,
                    "source": {
                        **nrel_source,
                        "locator": nrel_source["locator"] + f"&record={row_number}",
                        "attribution": nrel_source["attribution"]
                        + " CSV SHA-256 "
                        + table["csv_sha256"],
                    },
                }
            )
    # Read only documented name/alias fields; no PAZy sequence cache or historical identity proof.
    value["alias_source_checked"] = query is not None
    snapshot = load_activity_source()["result"] if query is not None else None
    if snapshot is not None:
        value["alias_source_available"] = True
        sources.append({**snapshot["source"], "locator": "#pinned-name-alias-scope"})
        for index, record in enumerate(snapshot["records"]):
            basis = [
                b
                for label, alias in [
                    (record["protein_name"], False),
                    *((s.get("name"), True) for s in record.get("synonyms") or []),
                ]
                if isinstance(label, str) and (b := name_basis(label, alias))
            ]
            if basis:
                candidates.append(
                    {
                        "source_name": "PAZy activity snapshot",
                        "identifier": str(record["protein_id"]),
                        "name": record["protein_name"],
                        "basis": list(dict.fromkeys(basis)),
                        "source": {
                            **snapshot["source"],
                            "locator": f"#/{index}",
                            "doi": record["literature_doi"],
                        },
                    }
                )
    elif query is not None:
        value["issues"].append("ALIAS_SOURCE_UNAVAILABLE")
    # A name plus a sequence identify only when they point at the same NREL row; otherwise neither is trusted.
    if query is not None and sequence is not None:
        agreement = bool(exact_rows & named_rows)
        value["name_sequence_agreement"] = "MATCH" if agreement else "UNVERIFIED"
        if not agreement:
            value["issues"].append("NAME_SEQUENCE_NOT_RECONCILED")
    else:
        agreement = True
    value["identity_status"] = (
        "IDENTIFIED_SEQUENCE"
        if exact_rows and agreement
        else "CANDIDATE_MATCH"
        if candidates
        else "UNRESOLVED"
    )
    candidates.sort(
        key=lambda c: (
            0 if "EXACT_SEQUENCE" in c["basis"] else 1 if "EXACT_FRAGMENT" in c["basis"] else 2
        )
    )
    value.update(
        candidates=candidates[:limit],
        total_candidates=len(candidates),
        omitted_candidates=max(0, len(candidates) - limit),
        multiple_source_records=len(candidates) > 1,
    )
    return outcome(
        "PARTIAL"
        if "ALIAS_SOURCE_UNAVAILABLE" in value["issues"]
        else "SUCCESS"
        if candidates
        else "NO_DATA",
        value,
        sources,
        LIMITATIONS,
        "COMPUTED",
    )
