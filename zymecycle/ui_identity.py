"""Presentation-only triage boundary: failed previews are not identity records."""

from __future__ import annotations

from typing import TypedDict

import streamlit as st

from .agent_types import ToolUpdate
from .privacy import identifier_safe, safe_public_text


class IdentityCandidateView(TypedDict):
    name: str
    basis: list[str]


class IdentityCard(TypedDict):
    tool_status: str
    identity_status: str | None
    sequence_validation: str | None
    sequence_length: int | None
    flagged_symbols: list[str]
    candidates: list[IdentityCandidateView]


def identity_card(update: ToolUpdate) -> IdentityCard | None:
    """Flow: Receives public ToolUpdate from render_dialogue_result(); returns optional IdentityCard for render_identity_update(), without filling missing scientific identity fields."""
    if update["name"] != "triage_protein":
        return None
    raw = update["data"].get("identity")
    payload = raw if isinstance(raw, dict) else {}
    status = payload.get("identity_status")
    validation = payload.get("sequence_validation")
    length = payload.get("sequence_length")
    symbols = payload.get("flagged_symbols")
    candidates = payload.get("nrel_candidates")
    projected: list[IdentityCandidateView] = []
    for item in candidates[:3] if isinstance(candidates, list) else []:
        if not isinstance(item, dict) or not identifier_safe(item.get("name")):
            continue
        name, basis = item.get("name"), item.get("basis")
        if isinstance(name, str):
            projected.append(
                {
                    "name": name,
                    "basis": [safe_public_text(b) for b in basis[:5] if isinstance(b, str)]
                    if isinstance(basis, list)
                    else [],
                }
            )
    return {
        "tool_status": safe_public_text(update["status"]),
        "identity_status": status
        if isinstance(status, str)
        and status in ("IDENTIFIED_SEQUENCE", "CANDIDATE_MATCH", "UNRESOLVED")
        else None,
        "sequence_validation": safe_public_text(validation)
        if isinstance(validation, str)
        else None,
        "sequence_length": length if type(length) is int and length >= 0 else None,
        "flagged_symbols": [s for s in symbols[:20] if isinstance(s, str) and len(s) == 1]
        if isinstance(symbols, list)
        else [],
        "candidates": projected,
    }


def render_identity_update(update: ToolUpdate) -> None:
    """Flow: Receives public ToolUpdate from render_dialogue_result(); returns None after bounded identity/status display via identity_card(), never changing established context."""
    card = identity_card(update)
    if card is None:
        return
    if card["tool_status"] == "ERROR":
        st.warning(
            "Local identity triage · ERROR. This call returned no usable identity evidence; established context is unchanged."
        )
        return
    if card["tool_status"] == "NO_DATA":
        st.info(
            "Local identity triage · NO_DATA. No source-backed match was returned; this does not prove the enzyme is absent."
        )
    if card["identity_status"] is None:
        st.warning(
            "Identity details are incomplete or unavailable. No identity status was inferred. Tool status: "
            + card["tool_status"]
        )
        return
    length = str(card["sequence_length"]) if card["sequence_length"] is not None else "unavailable"
    st.text(
        f"Local triage: {card['identity_status']} · {card['sequence_validation'] or 'validation unavailable'} · "
        f"length {length} · flagged {', '.join(card['flagged_symbols']) or 'none reported'}"
    )
    for candidate in card["candidates"]:
        st.text(
            "NREL label candidate: " + candidate["name"] + " · " + ", ".join(candidate["basis"])
        )
