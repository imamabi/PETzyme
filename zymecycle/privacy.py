"""Authoritative tool privacy policy and outbound screening, independent of agent/UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

Persistence = Literal["NREL_ONLY", "FORBIDDEN_PAZY_DERIVED"]
# Three deliberately different sensitivities: 35 for outbound screening (avoids redacting chemical
# names in transit), 19/20 at the input boundary (routes declared sequences to local triage).
# Coarse leakage heuristics, not biological identity or sequence-validity thresholds.
SEQUENCE_LIKE_RUN = 35
MAX_IDENTIFIER_ALPHA_RUN = 19
MIN_DECLARED_SEQUENCE_RUN = (
    20  # Stricter at the user-input boundary, not for source chemical names.
)
PUBLIC_TEXT_LIMIT = 300
FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "vector",
        "vectors",
        "embedding",
        "embeddings",
        "residue_embeddings",
        "last_hidden_state",
        "sequence",
        "normalized_sequence",
        "amino_acid_sequence",
        "sequence_sha256",
        "sequence_hash",
        "rows",
    }
)


@dataclass(frozen=True)
class ToolPrivacy:
    persistence: Persistence
    # Communication exclusion is a display boundary, not the persistence decision.
    communication_allowed: bool = True


PUBLIC_NREL = ToolPrivacy("NREL_ONLY")
TRANSIENT = ToolPrivacy("FORBIDDEN_PAZY_DERIVED")
LIVE_PROTEIN = ToolPrivacy("FORBIDDEN_PAZY_DERIVED", communication_allowed=False)
TOOL_PRIVACY: dict[str, ToolPrivacy] = {
    "load_nrel": PUBLIC_NREL,
    "parse_nrel": PUBLIC_NREL,
    "filter_observations": PUBLIC_NREL,
    "summarize_condition": PUBLIC_NREL,
    "select_nrel_protein": PUBLIC_NREL,
    "search_pazy_pet": LIVE_PROTEIN,
    "get_pazy_metadata": LIVE_PROTEIN,
    "get_associated_literature": LIVE_PROTEIN,
    "compare_identity": LIVE_PROTEIN,
    "assess_evidence": PUBLIC_NREL,
    "check_comparability": PUBLIC_NREL,
    "investigate_conditions": PUBLIC_NREL,
    "identify_evidence_gaps": PUBLIC_NREL,
    "find_similar_nrel_proteins": PUBLIC_NREL,
    "recommend_next_experiments": PUBLIC_NREL,
    "investigate_pet_evidence": TRANSIENT,
    "advise_next_experiment": TRANSIENT,
    "triage_protein": TRANSIENT,
    "discover_pet_enzymes": TRANSIENT,
}
EPHEMERAL_TOOLS = frozenset(
    name for name, policy in TOOL_PRIVACY.items() if policy.persistence != "NREL_ONLY"
)


def tool_privacy(name: str) -> ToolPrivacy:
    """Return the authoritative privacy policy for a tool (unknown tools → transient).

    Purpose: single source of truth mapping each registered tool to
        NREL_ONLY or FORBIDDEN_PAZY_DERIVED persistence plus a
        communication display flag. Fail-closed: unlisted names get
        transient handling rather than a permissive default.
    Flow: Receives tool name str from catalog/execution/presentation consumers; returns ToolPrivacy with conservative transient handling for unknown tools."""
    return TOOL_PRIVACY.get(name, TRANSIENT)


def sequence_like(text: str) -> bool:
    """Heuristically detect sequence-like text (FASTA header or 35+ letter run).

    Purpose: coarse leakage screen for outbound/request validation —
        deliberately not a biological classifier, so it may over-match;
        that direction is safe for a privacy boundary.
    Flow: Receives untrusted text str from request/presentation validation; returns bool for privacy screening, not biological inference."""
    return bool(
        re.search(r"^\s*>", text, re.M) or re.search(r"[A-Za-z]{%d,}" % SEQUENCE_LIKE_RUN, text)
    )


def identifier_safe(text: object) -> bool:
    """Check a label is safe for retained context: ASCII, bounded, no long runs or hashes.

    Purpose: keeps sequences, sequence-like runs (20+ letters), and
        SHA-256-looking strings out of anything retained or model-bound,
        while still allowing normal labels such as 'LCC-ICCG' or 'TEP 021'.
    Flow: Receives external label object from dialogue validation; returns bool before labels enter retained context or model input."""
    return (
        isinstance(text, str)
        and 0 < len(text) <= 120
        and text.isascii()
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _().+-]*", text) is not None
        and not re.search(r"[A-Za-z]{%d,}" % (MAX_IDENTIFIER_ALPHA_RUN + 1), text)
        and re.fullmatch(r"[0-9a-fA-F]{64}", text) is None
    )


def safe_public_text(value: object) -> str:
    """Render one value as bounded public text, withholding sequence-like content.

    Purpose: strips control characters, caps at 300 chars, and replaces
        anything sequence-like (or credential-marked) with a fixed
        notice. Used by every model/public projection.
    Flow: Receives untrusted source value object from evidence projection; returns bounded str with sequence-like and credential content withheld."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))[:PUBLIC_TEXT_LIMIT]
    return "[sequence-like text withheld]" if sequence_like(text) or "LLM|" in text else text


def outbound_safe(
    value: object,
    private_sequences: Iterable[str] = (),
    *,
    approved_arrays: Mapping[str, Sequence[float]] | None = None,
    approved_subset_arrays: Mapping[str, Sequence[float]] | None = None,
) -> bool:
    """Recursively screen a provider-bound payload for forbidden content.

    Purpose: blocks sequence-like strings, credential markers, any
        known private sequence (substring match on normalized text),
        forbidden fields (vectors, sequences, hashes, rows), and bare
        numeric arrays (embedding-shaped); JSON-looking strings are
        parsed and screened too. Approved numeric arrays (e.g. the
        dialogue's own temperatures) pass explicitly. Never retains
        values; returns a boolean only.
    Flow: Receives outgoing payload object and private sequences from run_investigation(); returns bool before provider invocation without retaining values."""
    if isinstance(value, str):
        normalized = "".join(value.split()).upper()
        if (
            sequence_like(value)
            or "LLM|" in value
            or any(s in normalized for s in private_sequences if s)
        ):
            return False
        if value.lstrip().startswith(("{", "[")):
            try:
                return outbound_safe(
                    json.loads(value),
                    private_sequences,
                    approved_arrays=approved_arrays,
                    approved_subset_arrays=approved_subset_arrays,
                )
            except (ValueError, RecursionError):
                return False
        return True
    if isinstance(value, dict):
        if FORBIDDEN_MODEL_FIELDS.intersection(value):
            return False
        for key, item in value.items():
            approved = (approved_arrays or {}).get(key)
            if (
                approved is not None
                and isinstance(item, list)
                and all(type(n) in (int, float) for n in item)
                and sorted(item) == sorted(approved)
            ):
                continue
            subset = (approved_subset_arrays or {}).get(key)
            if (
                subset is not None
                and isinstance(item, list)
                and 1 <= len(item) <= 2
                and all(type(n) in (int, float) for n in item)
                and set(item) <= set(subset)
            ):
                continue
            if not outbound_safe(
                item,
                private_sequences,
                approved_arrays=approved_arrays,
                approved_subset_arrays=approved_subset_arrays,
            ):
                return False
        return True
    if isinstance(value, list):
        return not (value and all(type(v) in (int, float) for v in value)) and all(
            outbound_safe(
                v,
                private_sequences,
                approved_arrays=approved_arrays,
                approved_subset_arrays=approved_subset_arrays,
            )
            for v in value
        )
    return True


def credential_like(text: str) -> bool:
    """Flow: Receives researcher input str; returns bool for question_contains_private_input() and dispatch_research_turn() before credential-shaped input is retained or processed."""
    return bool(re.search(r"\b(?:api[_ -]?key|password|secret|bearer|access[_ -]?token)\b\s*(?:is|:|=)\s*\S+", text, re.IGNORECASE)
                or re.search(r"\bsk-[A-Za-z0-9_-]{16,}\b", text))


def question_contains_private_input(text: str) -> bool:
    """Detect private input (sequences, FASTA, credential shapes) in a raw question.

    Purpose: stricter boundary than outbound_safe: any 20+ uppercase run
        or 'sequence:/FASTA:' declaration fails, so researcher sequences
        are routed to the transient local triage path instead of the
        model — while ordinary chemical terminology still passes.
    Flow: Receives external question str from interpret_request(); returns bool to route declared/sequence-shaped input away from the model while preserving public source terminology."""
    if not outbound_safe(text) or credential_like(text):
        return True
    if re.fullmatch(r"[A-Za-z]{%d,}" % MIN_DECLARED_SEQUENCE_RUN, text.strip()):
        return True
    if re.search(r"[A-Z]{%d,}" % MIN_DECLARED_SEQUENCE_RUN, text):
        return True
    if re.search(r"(?<![A-Za-z])[ACDEFGHIKLMNPQRSTVWYBJXZ]{20,}(?![A-Za-z])", text, re.IGNORECASE):
        return True
    return bool(
        re.search(r"(?:sequence|FASTA)\s*:\s*[A-Za-z]{%d,}" % MIN_DECLARED_SEQUENCE_RUN, text, re.I)
    )
