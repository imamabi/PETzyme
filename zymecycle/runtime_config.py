"""Finite runtime limits, separate from source provenance and provider credentials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

# Generous finite research defaults; cost guards remain independent of attempt allowances.
DEFAULT_TURN_MODELS = 16
DEFAULT_TURN_TOOLS = 16
# Absolute safety caps bound externally configured work; they are not scientific thresholds.
MAX_TURN_MODELS = 32
MAX_TURN_TOOLS = 32
MAX_SESSION_MODELS = 128
MAX_SESSION_TOOLS = 256
MAX_SESSION_COST_USD = 100.0
MAX_TOOL_CALLS_PER_MESSAGE = 8  # Existing provider envelope/batch memory bound.


class SessionUsage(TypedDict):
    actual_http_requests: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    unknown_usage_requests: int
    unknown_reasoning_requests: int
    reserved_cost_usd: float
    elapsed_seconds: float


def empty_usage() -> SessionUsage:
    """Return a zeroed SessionUsage accumulator for a new dialogue.

    Flow: Receives no external input from prepare_dialogue_turn(); returns zeroed SessionUsage for a new independent dialogue."""
    return {
        "actual_http_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "unknown_usage_requests": 0,
        "unknown_reasoning_requests": 0,
        "reserved_cost_usd": 0.0,
        "elapsed_seconds": 0.0,
    }


class RuntimeLimitValues(TypedDict):
    turn_models: int
    turn_tools: int
    session_models: int
    session_tools: int
    session_cost_usd: float


@dataclass(frozen=True)
class RuntimeLimits:
    turn_models: int = DEFAULT_TURN_MODELS
    turn_tools: int = DEFAULT_TURN_TOOLS
    session_models: int = 96
    session_tools: int = 128
    session_cost_usd: float = 10.0

    def __post_init__(self) -> None:
        for name, cap in (
            ("turn_models", MAX_TURN_MODELS),
            ("turn_tools", MAX_TURN_TOOLS),
            ("session_models", MAX_SESSION_MODELS),
            ("session_tools", MAX_SESSION_TOOLS),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(
                    "Runtime attempt limits must be finite positive integers within safety caps"
                )
        if self.session_models < self.turn_models or self.session_tools < self.turn_tools:
            raise ValueError("Session limits cannot be smaller than a full turn")
        if (
            type(self.session_cost_usd) not in (int, float)
            or not math.isfinite(self.session_cost_usd)
            or not 0 < self.session_cost_usd <= MAX_SESSION_COST_USD
        ):
            raise ValueError("Session cost ceiling must be finite and within the safety cap")

    def values(self) -> RuntimeLimitValues:
        """Flow: Receives RuntimeLimits configuration from agent/worker; returns RuntimeLimitValues for bounded model metadata and telemetry."""
        return {
            "turn_models": self.turn_models,
            "turn_tools": self.turn_tools,
            "session_models": self.session_models,
            "session_tools": self.session_tools,
            "session_cost_usd": self.session_cost_usd,
        }


def parse_runtime_limits(value: object) -> RuntimeLimits:
    """Validate an external worker configuration into finite RuntimeLimits.

    Purpose: rejects non-integer/out-of-cap attempts, session limits
        smaller than a turn, and non-positive/oversize spending
        ceilings before any inference can consume them.
    Raises:
        ValueError: on any invalid configuration.
    Flow: Receives external worker configuration object; returns validated RuntimeLimits or raises ValueError before inference."""
    if value is None:
        return RuntimeLimits()
    if not isinstance(value, dict) or set(value) != set(RuntimeLimitValues.__annotations__):
        raise ValueError("Invalid runtime configuration fields")
    return RuntimeLimits(**value)
