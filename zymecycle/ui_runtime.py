"""Researcher-configured finite runtime controls and aggregate-only UI accounting."""

from __future__ import annotations

from typing import TypedDict

import streamlit as st

from .meta_telemetry import accumulate_session_usage, estimated_token_cost
from .runtime_config import (
    MAX_SESSION_COST_USD,
    MAX_SESSION_MODELS,
    MAX_SESSION_TOOLS,
    MAX_TURN_MODELS,
    MAX_TURN_TOOLS,
    RuntimeLimits,
    SessionUsage,
    empty_usage,
)
from .ui_adapter import ResearchView


class RuntimeLedger(TypedDict):
    model_attempts: int
    tool_attempts: int
    usage: SessionUsage


class RuntimeReservation(TypedDict):
    model_attempts: int
    tool_attempts: int
    cost_usd: float


def runtime_ledger() -> RuntimeLedger:
    """Flow: Receives active Streamlit session metadata; returns RuntimeLedger for runtime_controls(), remaining_limits() and record_live_usage(), with no scientific records."""
    if "runtime_ledger" not in st.session_state:
        st.session_state["runtime_ledger"] = {
            "model_attempts": 0,
            "tool_attempts": 0,
            "usage": empty_usage(),
        }
    ledger: RuntimeLedger = st.session_state["runtime_ledger"]
    return ledger


def runtime_controls(prefix: str) -> RuntimeLimits:
    """Flow: Receives external UI configuration via main() or render_live_dialogue(); returns validated RuntimeLimits for remaining_limits() before any authorized launch."""
    defaults = RuntimeLimits()
    with st.expander("Execution limits · requests, tools and spending"):
        st.caption(
            "Model requests count reasoning/tool-selection steps. Completion tokens bound one "
            "response, including reasoning. Spending is a separate pre-HTTP estimate guard."
        )
        values = {}
        for name, label, cap in (
            ("turn_models", "Model requests per investigation/turn", MAX_TURN_MODELS),
            ("turn_tools", "Scientific tool attempts per investigation/turn", MAX_TURN_TOOLS),
            ("session_models", "Cumulative session model attempts", MAX_SESSION_MODELS),
            ("session_tools", "Cumulative session scientific attempts", MAX_SESSION_TOOLS),
        ):
            values[name] = int(
                st.number_input(
                    label,
                    min_value=1,
                    max_value=cap,
                    value=getattr(defaults, name),
                    step=1,
                    key=prefix + "_" + name,
                )
            )
        cost = st.number_input(
            "Cumulative session spending ceiling (estimated USD)",
            min_value=0.01,
            max_value=MAX_SESSION_COST_USD,
            value=defaults.session_cost_usd,
            key=prefix + "_session_cost",
        )
        ledger = runtime_ledger()
        st.caption(
            f"Session totals across live modes and resets: {ledger['model_attempts']} model / "
            f"{ledger['tool_attempts']} scientific attempts; conservative estimated USD "
            f"{ledger['usage']['reserved_cost_usd']:.6f}."
        )
        st.caption(
            "If a request limit is reached, increase it explicitly and authorize another submission. "
            "Dialogue continues from validated minimal context; a one-shot submission starts fresh "
            "and retrieves evidence again. No raw provider transcript is retained or silently replayed."
        )
        st.caption(
            "If worker accounting is unavailable, its worst-case allowance remains reserved; a context reset does not release that reservation."
        )
    return RuntimeLimits(**values, session_cost_usd=float(cost))


def submitted_limits(prefix: str) -> RuntimeLimits:
    """Flow: Receives submitted configuration widgets from submit_live_dialogue(); returns RuntimeLimits for the current turn before rendering the next form, avoiding stale prior-run configuration."""
    defaults = RuntimeLimits()
    return RuntimeLimits(
        **{
            name: st.session_state.get(prefix + "_" + name, getattr(defaults, name))
            for name in ("turn_models", "turn_tools", "session_models", "session_tools")
        },
        session_cost_usd=st.session_state.get(prefix + "_session_cost", defaults.session_cost_usd),
    )


def remaining_limits(config: RuntimeLimits) -> tuple[RuntimeLimits, float]:
    """Flow: Receives RuntimeLimits from runtime_controls(); returns remaining RuntimeLimits and float cost ceiling for run_keychain_investigation(), or raises before launch."""
    ledger = runtime_ledger()
    models = config.session_models - ledger["model_attempts"]
    tools = config.session_tools - ledger["tool_attempts"]
    cost = config.session_cost_usd - ledger["usage"]["reserved_cost_usd"]
    if min(models, tools) < 1:
        raise ValueError("SESSION_ATTEMPT_LIMIT: review cumulative limits before continuing.")
    if cost <= 0:
        raise ValueError("SPENDING_LIMIT: cumulative estimated ceiling reached; no request sent.")
    return RuntimeLimits(
        turn_models=min(config.turn_models, models),
        turn_tools=min(config.turn_tools, tools),
        session_models=config.session_models,
        session_tools=config.session_tools,
        session_cost_usd=config.session_cost_usd,
    ), cost


def reserve_live_usage(limits: RuntimeLimits, ceiling: float) -> RuntimeReservation:
    """Flow: Receives remaining RuntimeLimits/cost from remaining_limits(); returns RuntimeReservation for record_live_usage(), holding worst-case numeric allowances before a child launch or uncertain worker outcome."""
    ledger = runtime_ledger()
    reservation = RuntimeReservation(
        model_attempts=limits.turn_models, tool_attempts=limits.turn_tools, cost_usd=ceiling
    )
    ledger["model_attempts"] += reservation["model_attempts"]
    ledger["tool_attempts"] += reservation["tool_attempts"]
    ledger["usage"]["reserved_cost_usd"] += ceiling
    st.session_state["runtime_ledger"] = ledger
    return reservation


def record_live_usage(view: ResearchView, reservation: RuntimeReservation) -> None:
    """Flow: Receives ResearchView from run_keychain_investigation() and RuntimeReservation from reserve_live_usage(); returns None after reconciling numeric usage, retaining the reservation when real-provider accounting is unavailable."""
    if view["mode"] == "LIVE_META" and (
        "provider_telemetry" not in view or "provider_budget" not in view
    ):
        return
    ledger = runtime_ledger()
    ledger["model_attempts"] += view["model_requests"] - reservation["model_attempts"]
    ledger["tool_attempts"] += view["tool_attempts"] - reservation["tool_attempts"]
    ledger["usage"]["reserved_cost_usd"] -= reservation["cost_usd"]
    ledger["usage"] = accumulate_session_usage(
        ledger["usage"], view.get("provider_telemetry", []), view.get("execution_seconds", 0.0)
    )
    st.session_state["runtime_ledger"] = ledger


def render_provider_usage(view: ResearchView) -> None:
    """Flow: Receives transient ResearchView from _render_findings_details() or render_dialogue_result(); returns None after displaying allowlisted numeric provider accounting and canonical validation counts, without retaining messages or scientific records."""
    entries = view.get("provider_telemetry", [])
    st.caption(
        (
            "Actual Meta HTTP attempts"
            if view["mode"] == "LIVE_META"
            else "Mock transport HTTP attempts"
        )
        + f" this turn: {len(entries)} · investigation {view.get('execution_seconds')} s"
    )
    st.caption(
        f"Backend/model attempts {view['model_requests']} · top-level scientific attempts {view['tool_attempts']}. These differ from actual HTTP activity."
    )
    for e in entries:
        st.text(
            f"Request {e['request_number']} · allowance {e['completion_allowance']} · HTTP {e['http_status']} · finish {e['finish_reason']} · input {e['prompt_tokens']} · cached {e['cached_input_tokens']} · completion {e['completion_tokens']} · reasoning {e['reasoning_tokens']} · visible {e['visible_output_tokens']} · model match {e['model_matches']} · total {e['total_tokens']} · seconds {e['elapsed_seconds']} · request bytes {e['request_bytes']} · error {e['error_code']}"
        )
    cost = estimated_token_cost(entries)
    st.caption(
        "Estimated Standard token cost: "
        + ("unavailable" if cost is None else f"USD ${cost:.6f}")
        + ". Estimate only; not a verified credit deduction. Missing cache detail uses uncached pricing."
    )
    st.text("Canonical final selection: " + str(view.get("synthesis", {}).get("status")))
    counts = view.get("selection_counts")
    if counts:
        st.text(
            f"Canonical evidence counts · required {counts['required']} · selected {counts['selected']} · omitted {counts['omitted']}"
        )
    budget = view.get("provider_budget", {})
    if budget.get("limit_usd") is not None:
        st.text(f"Estimated spend ceiling USD {budget['limit_usd']} · blocked {budget['blocked']}")
