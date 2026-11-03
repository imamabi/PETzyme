"""Allowlisted numeric provider accounting; never provider messages or source records."""

from __future__ import annotations

from typing import TypedDict

from .runtime_config import SessionUsage

INPUT_USD_PER_MILLION = 1.25
CACHED_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 4.25


class RequestTelemetry(TypedDict):
    request_number: int
    completion_allowance: int
    http_status: int | None
    finish_reason: str | None
    prompt_tokens: int | None
    cached_input_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    visible_output_tokens: int | None
    total_tokens: int | None
    elapsed_seconds: float
    request_bytes: int
    model_matches: bool | None
    error_code: str | None


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 10_000_000 else None


def record_usage(entry: RequestTelemetry, envelope: object, expected_model: str) -> None:
    """Record allowlisted numeric usage from a provider envelope into telemetry.

    Purpose: copies only model match, finish reason, token counts
        (0..10M else None), cached/reasoning splits, and derived
        visible-output tokens. Messages, prompts, and completions are
        never retained.
    Flow: Receives transient provider envelope dict from MetaModelClient.complete(); updates RequestTelemetry with safe numeric fields for investigate_with_backend()."""
    if not isinstance(envelope, dict):
        return
    entry["model_matches"] = (
        envelope.get("model") == expected_model if "model" in envelope else None
    )
    choices = envelope.get("choices")
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        entry["finish_reason"] = (
            reason if reason in ("stop", "tool_calls", "length", "content_filter") else "UNKNOWN"
        )
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        entry[field] = _count(usage.get(field))
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    entry["cached_input_tokens"] = (
        _count(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None
    )
    entry["reasoning_tokens"] = (
        _count(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else None
    )
    completion, reasoning = entry["completion_tokens"], entry["reasoning_tokens"]
    if completion is not None and reasoning is not None and reasoning <= completion:
        entry["visible_output_tokens"] = completion - reasoning


def estimated_token_cost(entries: list[RequestTelemetry]) -> float | None:
    """Estimate USD cost of finished requests at documented Standard rates.

    Purpose: $1.25/M uncached input, $0.15/M cached input, $4.25/M
        completion. Returns None when any entry lacks counts or reports
        impossible cache detail; missing cache detail prices at the
        uncached rate. An estimate only, never a verified deduction.
    Flow: Receives RequestTelemetry list from MetaModelClient.complete(); returns estimated USD float or None for Streamlit main() using documented Standard rates."""
    total = 0.0
    for entry in entries:
        prompt, completion, cached = (
            entry["prompt_tokens"],
            entry["completion_tokens"],
            entry["cached_input_tokens"],
        )
        if prompt is None or completion is None:
            return None
        # Missing cache detail uses the conservative uncached rate, not an invented cache count.
        if cached is not None and cached > prompt:
            return None
        total += (
            (prompt - (cached or 0)) * INPUT_USD_PER_MILLION
            + (cached or 0) * CACHED_USD_PER_MILLION
            + completion * OUTPUT_USD_PER_MILLION
        ) / 1_000_000
    return total


def projected_token_cost(
    entries: list[RequestTelemetry], request_bytes: int, completion_allowance: int
) -> float:
    """Conservatively project total spend including one unsent request.

    Purpose: pre-HTTP spending guard input. Unknown usage reserves one
        input token per serialized byte plus the full completion
        allowance — deliberately pessimistic so the ceiling trips early
        rather than late.
    Flow: Receives prior RequestTelemetry and next serialized request size/allowance from MetaModelClient.complete(); returns conservative estimated USD float for its optional pre-HTTP spending guard."""

    def reserve(size: int, completion: int) -> float:
        # One input token per serialized byte is a deliberately conservative estimate, not billing proof.
        return (size * INPUT_USD_PER_MILLION + completion * OUTPUT_USD_PER_MILLION) / 1_000_000

    total = reserve(request_bytes, completion_allowance)
    for entry in entries:
        known = estimated_token_cost([entry])
        total += (
            known
            if known is not None
            else reserve(entry["request_bytes"], entry["completion_allowance"])
        )
    return total


def accumulate_session_usage(
    previous: "SessionUsage", entries: list[RequestTelemetry], elapsed_seconds: float
) -> "SessionUsage":
    """Add one turn's telemetry into cumulative numeric session usage.

    Purpose: sums requests, tokens, unknown-count requests, reserved
        cost, and elapsed seconds. Counts only — no messages retained.
    Flow: Receives prior numeric SessionUsage and per-turn telemetry from investigate_with_backend(); returns cumulative SessionUsage without retaining provider/source messages."""
    current = previous.copy()
    current["actual_http_requests"] += len(entries)
    for entry in entries:
        prompt, completion, reasoning = (
            entry["prompt_tokens"],
            entry["completion_tokens"],
            entry["reasoning_tokens"],
        )
        current["prompt_tokens"] += prompt or 0
        current["completion_tokens"] += completion or 0
        current["reasoning_tokens"] += reasoning or 0
        current["unknown_usage_requests"] += int(prompt is None or completion is None)
        current["unknown_reasoning_requests"] += int(reasoning is None)
    current["reserved_cost_usd"] += projected_token_cost(entries, 0, 0)
    current["elapsed_seconds"] += elapsed_seconds
    return current
