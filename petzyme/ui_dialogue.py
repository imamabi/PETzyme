"""Transient live-dialogue display; only minimal context and aggregate spending survive reruns."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from .agent_types import ToolUpdate
from .dialogue import validate_dialogue_context
from .dialogue_types import DialogueInput
from .meta_telemetry import projected_token_cost
from .runtime_config import parse_runtime_limits
from .ui_adapter import ResearchView
from .ui_decision import render_experimental_decision as render_experimental_decision
from .ui_identity import render_identity_update
from .ui_live import run_keychain_investigation
from .ui_runtime import (
    record_live_usage,
    remaining_limits,
    render_provider_usage,
    reserve_live_usage,
    runtime_controls,
    submitted_limits,
)


def clear_live_dialogue() -> None:
    """Flow: Receives reset/mode-change events from reset_result() or render_live_dialogue(); returns None after clearing context and input widgets, preserving aggregate spending."""
    authorization_keys = [k for k in st.session_state if k.startswith("dialogue_authorize_")]
    for key in (
        "live_dialogue_context",
        "dialogue_text",
        "dialogue_kind",
        "dialogue_new",
        "dialogue_decision_support",
        "dialogue_submit_pending",
        *authorization_keys,
    ):
        st.session_state.pop(key, None)
    if authorization_keys:
        st.session_state["dialogue_epoch"] = st.session_state.get("dialogue_epoch", 0) + 1
    if st.session_state.get("mode") == "Live Muse dialogue":
        # Explicit values also reset the browser's retained, unsubmitted form controls.
        st.session_state.update(
            dialogue_new=True,
            dialogue_decision_support=False,
            dialogue_kind="ANSWER",
            dialogue_text="",
        )


def render_dialogue_result(
    view: ResearchView, render_findings: Callable[[ResearchView], None]
) -> None:
    """Flow: Receives transient ResearchView from submit_live_dialogue(); returns None after displaying actual model decisions and local triage without result history."""
    dialogue = view.get("dialogue")
    if dialogue is None or dialogue["outcome"] == "SCIENTIFIC_FINDINGS":
        render_findings(view)
        return
    contract = dialogue.get("decision_contract")
    with st.expander("Technical decision validation"):
        if contract is not None and contract["original_schema_valid"] is not None:
            st.caption(
                f"Original decision accepted: {contract['original_accepted']} · "
                f"schema correction attempts: {contract['correction_attempts']} · "
                f"correction outcome: {contract['correction_status']}"
            )
            if contract["initial_issues"] or contract["final_issues"]:
                with st.expander("Bounded structural validation diagnostics"):
                    for stage, issues in (
                        ("initial", contract["initial_issues"]),
                        ("final", contract["final_issues"]),
                    ):
                        for issue in issues:
                            st.text(
                                f"{stage}: {issue['section']} index "
                                f"{issue['option_index'] if issue['section'] == 'option' else issue['preference_index']} · "
                                f"field {issue['field']} · {issue['code']} · "
                                f"expected {issue['expected']} · received type {issue['received_type']}"
                            )
    if dialogue["outcome"] == "EXPERIMENTAL_DECISION" and "decision" in dialogue:
        render_experimental_decision(dialogue["decision"], view["mode"])
    elif dialogue["outcome"] == "ASK_CLARIFICATION":
        st.info(dialogue["clarification"]["question"])
        if dialogue["clarification"]["presentation"] == "DETERMINISTIC_FALLBACK":
            st.caption(
                "Grounded deterministic fallback: the model intent was validated, but its unverified wording was withheld. Scientific status: PARTIAL."
            )
        else:
            st.caption(
                "Muse formulated this information request; scientific facts and intent were verified separately. Scientific status: PARTIAL."
                if view["mode"] == "LIVE_META"
                else "Mocked model decision; not real Muse acceptance. Scientific status: PARTIAL."
            )
    else:
        st.error(
            "Live Muse dialogue incomplete. No valid scientific answer or clarification was accepted."
        )
        st.write(view.get("synthesis", {}).get("reason"))
        if "decision" in dialogue["context"]:
            st.caption(
                "No adaptive experimental recommendation was accepted. Any executed component advice below is deterministic source-tool output, not a replacement for Muse's decision."
            )
        if view["mode"] != "LIVE_META":
            st.caption("Mocked model execution; this is not real Muse acceptance.")
    for update in view["progress"]:
        render_identity_update(update)
    with st.expander("Scientific limitations and source-grounded identity facts"):
        for fact in dialogue["identity_facts"]:
            st.text(fact["text"])
    with st.expander("Numeric provider usage"):
        render_provider_usage(view)
        for update in view["progress"]:
            st.text(
                f"{update.get('actor', 'MODEL')}: {update['name']} · {update['status']}"
                + (" · " + str(update["error_code"]) if update.get("error_code") else "")
            )
        if not view["progress"]:
            st.caption(
                "No model-selected scientific tool calls this turn. Any retained NREL source references are revalidated locally; PAZy records are not retained."
            )
        for receipt in view.get("source_coverage", []):
            st.caption(
                f"{receipt['source']}: {receipt['status']} · {receipt['measurements']} retrieved measurements · {receipt['reason']}"
            )
    if view.get("observations") or view.get("advice") or view.get("multi_study"):
        st.caption(
            "Executed scientific-tool findings below remain distinct from Muse’s clarification question; the original two-enzyme choice is unresolved."
        )
        render_findings(view)
    st.warning(
        "A source-listed label or exact reference sequence does not establish experimental construct identity or PET activity. No general two-enzyme ranking has been performed."
    )


def queue_live_dialogue() -> None:
    """Flow: Receives the submit widget event from render_live_dialogue(); returns None after setting one boolean for processing in its normal page run, without retaining inputs or results."""
    st.session_state["dialogue_submit_pending"] = True


def submit_live_dialogue(render_findings: Callable[[ResearchView], None]) -> None:
    """Flow: Receives external form inputs from render_live_dialogue(); returns None after one authorized worker turn and transient display via render_dialogue_result()."""
    text = st.session_state.pop("dialogue_text", "")
    epoch = st.session_state.get("dialogue_epoch", 0)
    authorized = st.session_state.pop("dialogue_authorize_" + str(epoch), False)
    st.session_state["dialogue_epoch"] = epoch + 1
    context = (
        None
        if st.session_state.get("dialogue_new", True)
        else st.session_state.get("live_dialogue_context")
    )
    try:
        limits = submitted_limits("dialogue")
    except ValueError as error:
        st.error(str(error))
        return
    spent = st.session_state.get("dialogue_spent", 0.0)
    if not authorized:
        st.error("Authorize this paid turn explicitly; no request was sent.")
        return
    if spent >= limits.session_cost_usd:
        st.error("Temporary acceptance-session estimate ceiling reached; no request was sent.")
        return
    try:
        active_limits, remaining_cost = remaining_limits(limits)
    except ValueError as error:
        st.error(str(error))
        return
    payload: DialogueInput = {
        "context": context,
        "text": text,
        "input_kind": st.session_state.get("dialogue_kind", "ANSWER"),
    }
    if st.session_state.get("dialogue_decision_support", False):
        payload["decision_support"] = True
    reservation = reserve_live_usage(
        active_limits, min(limits.session_cost_usd - spent, remaining_cost)
    )
    with st.status("Investigating your research question…", expanded=True) as progress:

        def observe(update: ToolUpdate) -> None:
            progress.write(update["name"] + " · " + update["status"] + " — " + update["summary"])

        result = run_keychain_investigation(
            "Local identity dialogue",
            authorized=True,
            completion_tokens=10000,
            dialogue_input=payload,
            cost_ceiling_usd=min(limits.session_cost_usd - spent, remaining_cost),
            runtime_limits=active_limits,
            on_update=observe,
        )
        returned = result["result"]
        accepted = (
            returned is not None and returned.get("synthesis", {}).get("status") != "INCOMPLETE"
        )
        progress.update(
            label="Scientific tools finished" if accepted else "Investigation incomplete",
            state="complete" if accepted else "error",
            expanded=not accepted,
        )
    view = result["result"]
    if view is None:
        st.error(result["message"])
        return
    record_live_usage(view, reservation)
    st.session_state["dialogue_spent"] = spent + projected_token_cost(
        view.get("provider_telemetry", []), 0, 0
    )
    if view.get("dialogue"):
        st.session_state["live_dialogue_context"] = view["dialogue"]["context"]
        st.session_state["dialogue_decision_support"] = "decision" in view["dialogue"]["context"]
        st.session_state["dialogue_new"] = False
        st.session_state["dialogue_kind"] = "ANSWER"
    render_dialogue_result(view, render_findings)


def render_live_dialogue(render_findings: Callable[[ResearchView], None]) -> None:
    """Flow: Receives rendering callback from main(); returns None after rendering minimal context and the explicit paid-turn submission form."""
    st.subheader("Live Muse scientific dialogue")
    st.caption(
        "Muse investigates general research questions through the production scientific agent. Structured enzyme-choice questions retain validated dialogue context; general investigations start fresh. Sequences remain local and results transient."
    )
    if st.session_state.pop("dialogue_submit_pending", False):
        st.markdown('<div id="latest-dialogue-findings"></div>', unsafe_allow_html=True)
        st.markdown("[Continue with a follow-up question ↓](#dialogue-question)")
        submit_live_dialogue(render_findings)
    state = st.session_state.get("live_dialogue_context")
    if state is not None and not validate_dialogue_context(state):
        st.warning(
            "The in-memory dialogue uses an older or invalid contract. Reset explicitly; aggregate spending will be preserved."
        )
        st.button("Reset live dialogue context", key="dialogue_reset", on_click=clear_live_dialogue)
        return
    if state:
        st.text("Original objective: choose between " + ", ".join(state["requested_labels"]))
        st.text(
            "Established NREL labels: "
            + ", ".join(n or "unresolved" for n in state["confirmed_labels"])
        )
        for requested, rejected in zip(state["requested_labels"], state["rejected_labels"]):
            if rejected:
                st.caption("Rejected for " + requested + ": " + ", ".join(rejected))
        st.text(
            "Available temperatures: "
            + ", ".join(f"{t:g}°C" for t in state["temperatures_c"])
            + f" · maximum pH {state['maximum_ph']:g} (not an assay setpoint)"
        )
        if state.get("assay_ph") is not None:
            st.text(
                f"Requested assay pH {state['assay_ph']} · PET {state['pet_form']} · explicit focus {state['focus_label']}"
            )
        limits = parse_runtime_limits(st.session_state.get("dialogue_runtime_limits"))
        with st.expander("Session usage and execution limits"):
            st.text(
                f"Cumulative session attempts: model {state['model_attempts']}/{limits.session_models} · scientific {state['tool_attempts']}/{limits.session_tools}; per turn {limits.turn_models}/{limits.turn_tools}"
            )
            st.caption(
                f"Recorded session usage: {state['usage']['actual_http_requests']} HTTP requests · input {state['usage']['prompt_tokens']} · completion {state['usage']['completion_tokens']} · conservative cost USD {state['usage']['reserved_cost_usd']:.6f}"
            )
        if state["pending_question"]:
            st.caption("Pending scientific question: " + state["pending_question"])
    st.markdown('<div id="dialogue-question"></div>', unsafe_allow_html=True)
    st.markdown("#### Your next question" if state else "#### Start a research question")
    if state:
        st.caption(
            "Your objective and constraints are retained. Results are transient. To narrow available equipment, try: I can only run experiments at 40C now."
        )
    with st.form("live_dialogue_form", clear_on_submit=False):
        try:
            configured_limits = runtime_controls("dialogue")
        except ValueError as error:
            st.error(str(error))
            return
        st.session_state["dialogue_runtime_limits"] = configured_limits.values()
        st.checkbox(
            "Start a new independent decision context", key="dialogue_new", value=state is None
        )
        st.checkbox(
            "Adaptive experimental decision support",
            key="dialogue_decision_support",
            value=None
            if "dialogue_decision_support" in st.session_state
            else bool(state and "decision" in state),
            disabled=state is not None,
        )
        st.caption(
            "Optional: Muse selects hypothetical alternatives and qualitative priority. Existing source instruments and Python validate evidence, controls and constraints. Reset the decision context before changing this option."
        )
        st.selectbox(
            "Dialogue input type",
            ["ANSWER", "SEQUENCE", "REFERENCE"],
            key="dialogue_kind",
            format_func=lambda k: {
                "ANSWER": "Question or clarification reply",
                "SEQUENCE": "Raw sequence / FASTA (local only)",
                "REFERENCE": "Accession / publication DOI",
            }[k],
        )
        st.text_area(
            "Dialogue turn",
            key="dialogue_text",
            max_chars=16384,
            placeholder="I can only run experiments at 40C now."
            if state
            else "Should I use ESM050 or LCC-ICCG? My equipment supports 40C and 60C, but I cannot exceed pH 7.",
        )
        epoch = st.session_state.get("dialogue_epoch", 0)
        st.checkbox(
            "Authorize one paid Muse dialogue turn (10,000 completion ceiling)",
            key="dialogue_authorize_" + str(epoch),
        )
        st.form_submit_button(
            "Send live dialogue turn",
            key="dialogue_send",
            on_click=queue_live_dialogue,
        )
    st.caption(
        "After identity clarification, an exact-condition reply can use: pH 6.5, crystalline PET powder; optionally prefix Please investigate SOURCE_LABEL at. Values outside retained device limits are rejected."
    )
    st.caption(
        "The configured cumulative spending ceiling applies across live modes and resets. No automatic retries or alternate models. Scientific findings are not saved as chat history."
    )
    st.button("Reset live dialogue context", key="dialogue_reset", on_click=clear_live_dialogue)
