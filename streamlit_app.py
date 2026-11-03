"""PETzyme research workspace with local evidence and secure Muse access."""

from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

import streamlit as st

from scripts.launch_meta import keychain_item_present
from zymecycle.agent_request import interpret_request
from zymecycle.agent_types import ToolUpdate
from zymecycle.clarification import parse_clarification_context
from zymecycle.dialogue_types import DialogueInput
from zymecycle.meta import COMPLETION_ALLOWANCES, MAX_COMPLETION_TOKENS, valid_completion_allowance
from zymecycle.privacy import credential_like, question_contains_private_input
from zymecycle.research_session import ResearchSession, new_research_session, record_turn_status
from zymecycle.runtime_config import DEFAULT_TURN_MODELS, DEFAULT_TURN_TOOLS, RuntimeLimits
from zymecycle.ui_adapter import (
    EXAMPLES,
    ROOT,
    ResearchView,
    dispatch_research_turn,
    run_offline,
    runtime_identity,
)
from zymecycle.ui_advisor import render_research_advice
from zymecycle.ui_clarification import clear_clarification, render_clarification
from zymecycle.ui_communication import render_communication
from zymecycle.ui_dialogue import clear_live_dialogue, render_dialogue_result, render_live_dialogue
from zymecycle.ui_findings import display_response, finding_answer, prioritize_limitations
from zymecycle.ui_live import run_keychain_investigation
from zymecycle.ui_multistudy import render_multi_study
from zymecycle.ui_pazy import ARTICLE_DOI, PAZY_URL, discover_pazy_pet
from zymecycle.ui_results import focus_result, render_result
from zymecycle.ui_runtime import (
    record_live_usage,
    remaining_limits,
    render_provider_usage,
    reserve_live_usage,
    runtime_controls,
    submitted_limits,
)
from zymecycle.unit_context import unit_context


def reset_result() -> None:
    """Flow: Receives external mode-change callback; returns None after clearing the prior public view and per-investigation authorization."""
    st.session_state.pop("view", None)
    st.session_state.pop("error", None)
    st.session_state.pop("focus_findings", None)
    st.session_state.pop("pending_oneshot", None)
    clear_clarification()
    clear_live_dialogue()
    st.session_state["auth_epoch"] = st.session_state.get("auth_epoch", 0) + 1


class QueuedInvestigation(TypedDict):
    question: str
    mode: Literal["Offline demonstration", "Live Meta agent"]
    authorized: bool
    allowance: int
    limits: RuntimeLimits


def queue_investigation() -> None:
    """Flow: Receives submitted public question, authorization and runtime widgets from main(); returns None after queuing one validated QueuedInvestigation for main(), clearing prior consent without retaining scientific results or credentials."""
    epoch = st.session_state.get("auth_epoch", 0)
    authorized = st.session_state.get(f"authorize_{epoch}", False) is True
    st.session_state.pop(f"authorize_{epoch}", None)
    st.session_state["auth_epoch"] = epoch + 1
    st.session_state.pop("view", None)
    st.session_state.pop("error", None)
    st.session_state.pop("pending_oneshot", None)
    try:
        question = st.session_state["question"]
        interpret_request(question)  # No private sequence/key content enters the queue.
        mode = st.session_state["mode"]
        if mode not in ("Offline demonstration", "Live Meta agent"):
            raise ValueError("Choose a supported investigation mode.")
        allowance = st.session_state.get("completion_allowance", MAX_COMPLETION_TOKENS)
        if not valid_completion_allowance(allowance):
            raise ValueError("Select a supported completion allowance.")
        pending: QueuedInvestigation = {
            "question": question, "mode": mode, "authorized": authorized,
            "allowance": allowance,
            "limits": submitted_limits("oneshot") if mode == "Live Meta agent" else RuntimeLimits(),
        }
        st.session_state["pending_oneshot"] = pending
    except ValueError as error:
        st.session_state["error"] = str(error)


def select_example(question: str) -> None:
    """Flow: Receives external example-button question str; returns None after preparing main() input without executing an investigation."""
    reset_result()
    st.session_state["question"] = question


def render_findings(view: ResearchView) -> None:
    """Flow: Receives ResearchView from present_investigation() via main(); returns None after render_result() places the complete _render_findings_details() below a concise public overview."""
    if view.get('runtime') and view['runtime']['code_fingerprint'] != runtime_identity()['code_fingerprint']:
        st.error('The server and investigation loaded different code versions. Restart the local application before continuing.')
    render_result(view, lambda: _render_findings_details(view))
    if st.session_state.pop("focus_findings", False):
        focus_result()


def _render_findings_details(view: ResearchView) -> None:
    """Flow: Receives ResearchView from render_findings(); returns None after rendering complete findings and diagnostics inside render_result()'s collapsed detail panel."""
    scope = view.get("scientific_scope")
    if view.get('runtime'):
        st.caption(f"Loaded code {view['runtime']['code_fingerprint']} · worker process {view['runtime']['process_id']}")
    if view.get('interpretation'):
        st.caption('Request interpretation: ' + view['interpretation']['status'])
    if view.get('source_coverage'):
        st.markdown('#### Source coverage receipts')
        for receipt in view['source_coverage']:
            st.caption(f"{receipt['source']}: {receipt['status']} · {receipt['measurements']} retrieved measurements · {receipt['reason']}")
    if view['progress']:
        with st.expander('Executed scientific tools and ownership'):
            for update in view['progress']:
                st.text(f"{update.get('actor','MODEL')}: {update['name']} · {update['status']}" + (" · " + str(update["error_code"]) if update.get("error_code") else ""))
    if view["observations"]:
        units = unit_context(json.loads((ROOT / "NREL_SOURCE_MANIFEST.json").read_text()))
        st.caption(units["publication"])
        st.caption(units["record_verification"])
        with st.expander("Publication, workbook and condensed CSV units"):
            st.write(units["workbook"])
            st.write(units["csv_inference"])
            st.link_button("Original ACS Catalysis study", units["source_url"])
    if scope and scope["request"]["purpose"] == "EXPERIMENT_PLANNING":
        st.info("Exploratory planning · clarify candidate conditions; no established optimum.")
        if view.get("planning_question"):
            st.info(view["planning_question"])
            st.caption(
                "Source-grounded deterministic clarification; Muse selects the evidence, not this wording."
            )
        for explanation in scope["explanations"]:
            st.write(explanation)
    if scope and scope["background_only"]:
        st.warning(
            "Scientific scope: evidence below is background only, not direct support for the requested comparison."
        )
        for explanation in scope["explanations"]:
            st.write(explanation)
    synthesis = view.get("synthesis")
    if synthesis and synthesis.get("reason_code") in (
        "MODEL_REQUEST_LIMIT",
        "SCIENTIFIC_TOOL_ATTEMPT_LIMIT",
        "SPENDING_LIMIT",
        "COMPLETION_LENGTH_LIMIT",
    ):
        st.warning(str(synthesis["reason_code"]) + ": " + str(synthesis.get("reason")))
        st.caption(
            "No automatic retry. Review the corresponding execution setting before an explicitly authorized continuation; a fresh one-shot submission does not resume a provider transcript."
        )
    if view.get('routing'):
        st.text('Interpreted intent: ' + ', '.join(view['routing']['objectives']))
        st.text('Requested source scope: ' + ', '.join(view['routing']['sources']))
        st.text('Tools exposed to Muse: ' + ', '.join(view['routing']['exposed_tools']))
    if "provider_telemetry" in view:
        with st.expander("Provider request usage and timing"):
            render_provider_usage(view)
    if synthesis and synthesis["status"] == "INCOMPLETE":
        st.error("Live Muse synthesis incomplete.")
        if view["mode"] != "LIVE_META":
            st.caption("Offline/mocked execution; this is not real Muse acceptance.")
        st.write("Why synthesis stopped: " + synthesis["reason"])
        st.caption(
            "The evidence and deterministic tool conclusions below remain separate from an accepted Muse answer."
        )
    if view.get("communication"):
        render_communication(view["communication"], view["mode"])
    if view.get("advice"):
        render_research_advice(view)
        return
    if view.get("multi_study"):
        render_multi_study(view)
        return
    st.divider()
    st.subheader("Complete research findings")
    st.caption(view["question"])
    st.info(finding_answer(view))
    primary_limits, other_limits = prioritize_limitations(view)
    st.markdown("#### Scientific limitations")
    for limit in primary_limits:
        st.write("• " + limit)
    if other_limits:
        with st.expander(f"All remaining limitations ({len(other_limits)})"):
            for limit in other_limits:
                st.write("• " + limit)
    with st.expander("Execution details"):
        a, b, c = st.columns(3)
        a.metric("Workflow", view["workflow_status"])
        b.metric(
            "Backend attempts",
            f"{view['model_requests']} / {view.get('limits', {}).get('turn_models', DEFAULT_TURN_MODELS)}",
        )
        c.metric(
            "Scientific tool attempts",
            f"{view['tool_attempts']} / {view.get('limits', {}).get('turn_tools', DEFAULT_TURN_TOOLS)}",
        )
    st.caption(
        "COMPLETE means the workflow finished; it does not establish sufficient scientific evidence or activity superiority."
    )
    if view["mode"] == "DETERMINISTIC_OFFLINE":
        st.caption(
            "Offline backend attempts are deterministic orchestration steps. Live Muse requests: 0."
        )
    else:
        st.caption(
            "Backend attempts are reported by the workflow. This view does not expose independently observed HTTP counts or token usage."
        )
    for issue in view["issues"]:
        if view.get('discovery'):
            st.caption(issue)
        else:
            st.warning(issue)

    if view["observations"]:
        st.markdown("#### Recorded responses · MEASURED")
        forms = list(dict.fromkeys(o["pet_form"] for o in view["observations"]))
        for form in forms:
            st.caption(form.replace("_", " ").title() + " · experimental group not verified")
            values = [
                {
                    "Enzyme": o["enzyme_identifier"],
                    "Temperature (°C)": o["temperature_c"],
                    "pH": o["ph"],
                    "Recorded response": display_response(o["original_text"]),
                    "Units": o["unit_status"],
                }
                for o in view["observations"]
                if o["pet_form"] == form
            ]
            st.dataframe(
                values,
                hide_index=True,
                width="stretch",
                column_config={
                    "Recorded response": st.column_config.TextColumn("Recorded response (display)")
                },
            )
        st.caption(
            "Display rounded to six significant digits only; exact original strings and source cells are in provenance. No unit conversion."
        )
    missing = [c for c in view["conditions"] if c["status"] == "MISSING"]
    if missing:
        st.markdown("#### Missing exact-condition evidence")
        st.dataframe(
            [
                {
                    "Temperature (°C)": c["condition"]["temperature_c"],
                    "pH": c["condition"]["ph"],
                    "PET form": c["condition"]["pet_form"].replace("_", " ").title(),
                    "Observation": "MISSING — not zero",
                }
                for c in missing
            ],
            hide_index=True,
            width="stretch",
        )
    if view["gaps"]:
        st.markdown("#### Evidence still needed · COMPUTED")
        for gap in view["gaps"]:
            st.write(f"{gap['priority']}. {gap['requirement']}")
            for target in gap["target_conditions"]:
                st.caption(
                    f"Target: {target['temperature_c']}°C · pH {target['ph']} · {target['pet_form'].replace('_', ' ').title()}"
                )
    if view["similarity"]:
        similarity = view["similarity"]
        coverage = similarity["coverage"]
        st.markdown("#### Representation neighbors · COMPUTED")
        st.caption(
            f"Query: {similarity['query_identifier']} · searched {coverage['searched_candidates']} · "
            f"eligible {coverage['eligible_candidates']} · excluded in window {coverage['excluded_in_window']} · "
            f"outside window {coverage['outside_window_records']}"
        )
        st.warning(
            "The entire dataset was not searched. Cosine similarity is not PET activity, sequence identity or demonstrated functional equivalence."
        )
        st.dataframe(
            [
                {
                    "Rank": i,
                    "Protein": n["identifier"],
                    "NREL record": n["source_row"],
                    "Cosine similarity": n["cosine_similarity"],
                }
                for i, n in enumerate(similarity["neighbors"], 1)
            ],
            hide_index=True,
            width="stretch",
            column_config={"Cosine similarity": st.column_config.NumberColumn(format="%.6f")},
        )
    if view["plan"]:
        plan = view["plan"]
        st.markdown("#### Next investigations · PROPOSAL / HYPOTHETICAL")
        st.caption(
            f"{len(plan['recommendations'])} of {plan['candidate_count']} actions shown; {plan['omitted_count']} omitted. These are plans, not completed experiments."
        )
        priorities = {r["action_id"]: str(r["priority"]) for r in plan["recommendations"]}
        for action in plan["recommendations"]:
            with st.container(border=True):
                st.caption(action["category"].replace("_", " ") + " · HYPOTHETICAL")
                st.markdown(
                    f"**{action['priority']:02d} · {action['code'].replace('_', ' ').capitalize()}**"
                )
                st.write(action["proposed_action"])
                deps = (
                    ", ".join(
                        priorities[r["action_id"]]
                        for r in plan["recommendations"]
                        if r["action_id"] in action["prerequisites"]
                    )
                    or "None"
                )
                st.caption(
                    f"Feasibility: {action['feasibility']['status']} · Prerequisite actions: {deps}"
                )
                st.write("Information gained: " + action["expected_information"])
                with st.expander("Evidence and planning details"):
                    st.write("Objective: " + action["objective"])
                    for basis in action["evidence_basis"]:
                        st.write(f"[{basis['category']}] {basis['statement']}")
                        for source in basis["sources"]:
                            st.caption(source["locator"])
                    st.write("Required measurement or verification")
                    for requirement in action["required_measurements"]:
                        st.write("• " + requirement)
                    st.write(
                        "Controlled variables: "
                        + (
                            "; ".join(action["controlled_variables"])
                            or "No assay specified for this verification action."
                        )
                    )
                    st.write(
                        "Varied variables: "
                        + ("; ".join(action["varied_variables"]) or "None specified.")
                    )
                    st.write(
                        "Unknown requirements: "
                        + "; ".join(action["feasibility"]["unknown_requirements"])
                    )
                    for blocker in action["feasibility"]["blockers"]:
                        st.warning(blocker)
                    st.caption(action["ranking_rationale"])
                    st.json(
                        {
                            "action_id": action["action_id"],
                            "prerequisites": action["prerequisites"],
                            "target_conditions": action["target_conditions"],
                        }
                    )

    with st.expander("Scientific provenance and exact source locators"):
        st.caption(
            "Conditions and source cells are preserved. No response units have been converted."
        )
        for source in view["provenance"]:
            st.write(f"{source['name']} · {source['attribution']}")
            st.write(source["url"])
            if source["doi"]:
                st.caption("DOI: " + source["doi"])
        if view["comparison"]:
            st.json(view["comparison"])
        if view["record_identity"]:
            st.json(view["record_identity"])
        if view["observations"]:
            st.json(view["observations"])
        if view["observations"]:
            audit = json.loads((ROOT / "NREL_SOURCE_MANIFEST.json").read_text()).get(
                "unit_reconciliation"
            )
            if audit:
                st.write("NREL unit reconciliation: " + audit["interpretation"])
                st.json(audit)
        if view["conditions"]:
            st.json(view["conditions"])
        if view["similarity"]:
            st.json(
                {
                    k: view["similarity"][k]
                    for k in (
                        "query_source_row",
                        "csv_sha256",
                        "model",
                        "coverage",
                        "devices_used",
                        "fallbacks",
                        "neighbors",
                    )
                }
            )
    with st.expander("Executed scientific functions"):
        for update in view["progress"]:
            st.write(f"`{update['name']}` · {update['status']} — {update['summary']}")


def render_pazy_discovery() -> None:
    """Flow: Receives external PAZy search submission; renders transient ToolResult[list[ProteinView]] from discover_pazy_pet() without storing records in session state."""
    st.subheader("Explore PET-associated enzymes in PAZy")
    st.write(
        "Find biological annotations and explicitly associated publications through the live PAZy API. No Meta inference is used."
    )
    st.caption(
        "Biological annotations · not quantitative PET activity measurements. Results disappear on the next interaction; no history or export."
    )
    st.link_button("Official PAZy resource", PAZY_URL)
    st.link_button(
        "PAZy database article · Nucleic Acids Research (2026)", "https://doi.org/" + ARTICLE_DOI
    )
    with st.form("pazy_discovery", enter_to_submit=False):
        limit = st.selectbox("Maximum proteins", [1, 2, 3], index=2)
        submitted = st.form_submit_button("Search live PAZy", key="pazy_search")
    if not submitted:
        return
    with st.spinner("Retrieving one bounded live PET-protein page…"):
        result = discover_pazy_pet(limit)
    if result["result"] is None:
        st.error(result["message"])
        return
    st.caption(
        f"PAZy · {result['status']} · {len(result['result'])} proteins returned; at most one page, not an exhaustive catalog."
    )
    if not result["result"]:
        st.info("No proteins returned by this bounded search.")
    for protein in result["result"]:
        with st.container(border=True):
            st.text(protein["name"] or "Protein name unavailable")
            st.text("Source organism: " + (protein["organism"] or "Unavailable"))
            st.text(
                "Taxonomy: "
                + (protein["domain"] or "Unknown domain")
                + " / "
                + (protein["phylum"] or "Unknown phylum")
            )
            for association in protein["pet_associations"]:
                st.text("PAZy PET association: " + association)
            st.markdown("**Explicitly associated publications**")
            for publication in protein["publications"]:
                st.text(
                    f"{publication['title'] or 'Title unavailable'} · {publication['author'] or 'Authors unavailable'} · {publication['year'] or 'Year unavailable'}"
                )
                if publication["doi_url"]:
                    st.link_button("Publication DOI", publication["doi_url"])
            counts = protein["reference_counts"]
            st.caption(
                f"References: total {counts['total'] if counts['total'] is not None else 'unknown'}, returned {counts['returned']}, omitted {counts['omitted'] if counts['omitted'] is not None else 'unknown'}, unresolved among returned {counts['unresolved_returned']}."
            )
            for missing in protein["unavailable"]:
                st.text(missing)
    for limit in result["limitations"]:
        st.caption(limit)
    st.caption(
        "Source: PAZy, https://www.pazy.eu/ · database article DOI 10.1093/nar/gkag1026. No NREL identity match has been attempted."
    )


def legacy_main() -> None:
    """Flow: Receives external Streamlit widget events; invokes run_offline() or run_keychain_investigation() once per submission and renders only ResearchView through render_findings()."""
    st.set_page_config(page_title="PETzyme · Research workspace", page_icon="🔬", layout="wide")
    st.markdown(
        """<style>
    .stMainBlockContainer {max-width:1120px; padding-top:4.5rem; padding-bottom:3rem;}
    h1 {letter-spacing:-.035em;} h3 {letter-spacing:-.02em;}
    [data-testid="stMetric"] {background:#eaf1ed; border:1px solid #d8e5dd; border-radius:10px; padding:14px;}
    [data-testid="stSidebar"] {border-right:1px solid #d8e5dd;}
    </style>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("PETzyme")
        st.caption("PET HYDROLASE RESEARCH")
        st.divider()
        mode = st.radio(
            "Research mode",
            [
                "Offline demonstration",
                "Live Meta agent",
                "Live PAZy discovery",
                "Scientific clarification",
                "Live Muse dialogue",
            ],
            key="mode",
            on_change=reset_result,
        )
        if mode == "Offline demonstration":
            st.info(
                "Deterministic tool orchestration. Real local scientific evidence. No live Muse reasoning or paid inference."
            )
        elif mode == "Live Meta agent":
            st.info(
                "Muse selects scientific tools. Explicit authorization and a new Investigate action are required for each investigation."
            )
            st.caption(
                "Credentials are retrieved by the existing macOS Keychain launcher only after authorization. No key entry is needed here."
            )
        elif mode == "Live Muse dialogue":
            st.info(
                "Live Muse tool decisions and validated clarification. Every turn requires explicit paid authorization."
            )
        elif mode == "Scientific clarification":
            st.info(
                "Retain an enzyme-choice objective and equipment limits while clarifying identity locally. No paid inference."
            )
        else:
            st.info(
                "One small public PAZy request per search. Transient annotations only; no paid inference or persistent record history."
            )
        with st.expander("Which mode should I use?"):
            st.caption(
                "Offline demonstration: explore local evidence with guided examples. Scientific clarification: resolve names locally. Live Muse dialogue: investigate, plan and ask follow-ups. Live Meta agent: one question at a time. PAZy discovery: transient biological annotations."
            )
        st.divider()
        st.markdown("**Scientific sources**")
        st.caption(
            "NREL · DOI 10.5281/zenodo.15417757\n\nLocal ESM-2 · bounded representation search\n\nPAZy · pinned publication-derived activity curation; live protein annotations in discovery mode"
        )
        st.caption(
            "Saved research summaries withhold PAZy records. Live discovery displays annotations transiently. No raw investigation export or persistent scientific cache."
        )
        with st.expander('Loaded application version'):
            identity = runtime_identity()
            st.caption(f"Code {identity['code_fingerprint']} · process {identity['process_id']}")
    st.caption("PETZYME / RESEARCH WORKSPACE")
    st.title("Follow the evidence.")
    st.write(
        "Investigate PET hydrolases, inspect source evidence and plan the next useful experiment."
    )
    if mode == "Live Muse dialogue":
        render_live_dialogue(render_findings)
        return
    if mode == "Scientific clarification":
        render_clarification()
        return
    if mode == "Live PAZy discovery":
        render_pazy_discovery()
        return
    labels = [
        "40°C vs 60°C",
        "Missing: 50°C, pH 8",
        "ESM-2 neighbors",
        "Next experiment",
        "Multi-Study Evidence",
        "Outside NREL evidence",
        "Multi-study comparison",
        "Evidence → next experiment",
    ]
    with st.expander("Start with an example · eight options", expanded=False):
        columns = st.columns(2)
        for i, ((name, question), label) in enumerate(zip(EXAMPLES.items(), labels)):
            columns[i % 2].button(
                label,
                help=question,
                key=f"example_{i}",
                on_click=select_example,
                args=(question,),
                width="stretch",
            )
    if "question" not in st.session_state:
        st.session_state["question"] = next(iter(EXAMPLES.values()))
    transient_view: ResearchView | None = None
    pending = st.session_state.pop("pending_oneshot", None)
    if pending is not None:
        question = pending["question"]
        authorized = pending["authorized"]
        allowance = pending["allowance"]
        configured_limits = pending["limits"]
        st.session_state.pop("view", None)
        st.session_state.pop("error", None)
        with st.status("Investigation running", expanded=True) as progress:

            def observe(update: ToolUpdate) -> None:
                progress.write(f"`{update['name']}` · {update['status']} — {update['summary']}")

            try:
                live_kwargs = {}
                local_discovery = ('enzyme_discovery' in interpret_request(question)['objectives']
                                   and (mode == 'Offline demonstration' or not authorized))
                if mode == "Live Meta agent" and authorized and not local_discovery:
                    limits, ceiling = remaining_limits(configured_limits)
                    live_kwargs = {"runtime_limits": limits, "cost_ceiling_usd": ceiling}
                    reservation = reserve_live_usage(limits, ceiling)
                result = (
                    run_offline(question, on_update=observe)
                    if mode == "Offline demonstration" or local_discovery
                    else run_keychain_investigation(
                        question,
                        authorized=authorized,
                        on_update=observe,
                        completion_tokens=allowance,
                        **live_kwargs,
                    )
                )
                if result["result"] is not None:
                    st.session_state["focus_findings"] = True
                    if mode == "Live Meta agent" and not local_discovery:
                        record_live_usage(result["result"], reservation)
                    if result['result'].get('persistence','FORBIDDEN_PAZY_DERIVED') != 'NREL_ONLY' or result['result'].get('communication',{}).get('status') in ('VERIFIED_EXTRACTIVE','VERIFIED_BACKGROUND_EXPLANATION'):
                        transient_view = result["result"]
                    else:
                        st.session_state["view"] = result["result"]
                else:
                    st.session_state["error"] = (
                        result["message"] or "Investigation did not return public evidence."
                    )
            except ValueError as error:
                st.session_state["error"] = str(error)
            except Exception:
                st.session_state["error"] = (
                    "Investigation stopped. No substitute evidence was generated; no automatic retry was made."
                )
            progress.update(
                label="Investigation finished"
                + (" · " + transient_view["workflow_status"] if transient_view else ""),
                state="error"
                if transient_view
                and transient_view.get("synthesis", {}).get("status") == "INCOMPLETE"
                else "complete"
                if transient_view or "view" in st.session_state
                else "error",
            )
    epoch = st.session_state.get("auth_epoch", 0)
    with st.form("investigation", enter_to_submit=False):
        question = st.text_area(
            "Research question",
            key="question",
            height=105,
            max_chars=4000,
            placeholder="Ask which PET enzymes have the highest recorded responses, or choose a guided example.",
        )
        authorized = False
        allowance = MAX_COMPLETION_TOKENS
        if mode == "Live Meta agent":
            try:
                configured_limits = runtime_controls("oneshot")
            except ValueError as error:
                st.error(str(error))
                return
            allowance = st.selectbox(
                "Completion allowance per model request",
                (MAX_COMPLETION_TOKENS, *COMPLETION_ALLOWANCES),
                index=1,
                key="completion_allowance",
            )
            authorized = st.checkbox(
                "Authorize paid Meta inference for this investigation", key=f"authorize_{epoch}"
            )
        else:
            st.caption(
                "Offline mode supports the eight examples, local NREL + PAZy landscapes for supported identifiers, and broad PET-enzyme discovery. No paid inference."
            )
        st.form_submit_button("Investigate", type="primary", key="investigate", on_click=queue_investigation)
    if transient_view is not None:
        render_findings(transient_view)
        return
    if "error" in st.session_state:
        st.error(st.session_state["error"])
    if "view" in st.session_state:
        st.markdown("[View current findings ↓](#research-findings)")
        render_findings(st.session_state["view"])
    else:
        st.divider()
        st.caption(
            "Evidence first. Missing measurements stay missing; sequence similarity stays computational; experiment plans stay hypothetical."
        )


def queue_unified_turn() -> None:
    """Flow: Receives external submission widgets; returns None after queuing one privacy-checked turn for main()."""
    epoch = st.session_state.get("unified_epoch", 0)
    authorized = os.environ.get("PETZYME_OFFLINE") != "1" and keychain_item_present()
    st.session_state["unified_epoch"] = epoch + 1
    text = st.session_state.pop("unified_question", "")
    st.session_state["unified_question"] = ""
    st.session_state.pop("unified_error", None)
    try:
        if credential_like(text):
            st.session_state["unified_error"] = "Credential-like input was withheld; use the secure credential store."
            return
        if question_contains_private_input(text):
            session = st.session_state.get("research_session", new_research_session())
            updated, result, _ = dispatch_research_turn(session, text, run_offline)
            st.session_state["research_session"] = record_turn_status(updated, result["status"])
            identity = result["result"]
            if identity is None or "identity_status" not in identity:
                st.warning("Local sequence triage did not return an identity result.")
                return
            st.info("Local sequence triage: " + identity["identity_status"] + " · validation: "
                    + identity["sequence_validation"] + " · length: " + str(identity["sequence_length"])
                    + " · flagged: " + (", ".join(identity["flagged_symbols"]) or "none"))
            if updated.sequence_label:
                st.write("Exact authenticated NREL sequence match: " + updated.sequence_label)
            st.caption("Sequence matching does not establish experimental construct identity, PET activity or function. The sequence is not retained.")
            return
        interpret_request(text)
        st.session_state["unified_pending"] = {
            "question": text,
            "authorized": authorized,
        }
    except ValueError as error:
        st.session_state["unified_error"] = str(error)


def reset_unified_session() -> None:
    """Flow: Receives an external reset action; returns None after clearing research context while retaining runtime usage."""
    st.session_state["research_session"] = new_research_session()
    for key in ("unified_pending", "unified_error"):
        st.session_state.pop(key, None)
    st.session_state["unified_question"] = ""


def main() -> None:
    """Flow: Receives researcher inputs from Streamlit; returns None after dispatching and rendering the active research turn."""
    st.set_page_config(page_title="PETzyme · Research", page_icon="🔬", layout="wide")
    st.markdown("""<style>
    .stApp { color: var(--text-color); }
    .stApp .block-container { max-width: 68.75rem; width: 100%; min-width: 0; }
    .stApp [data-testid="stVerticalBlock"], .stApp [data-testid="stHorizontalBlock"],
    .stApp [data-testid="column"], .stApp [data-testid="stElementContainer"] { min-width: 0; }
    .stApp p, .stApp li, .stApp label, .stApp textarea, .stApp button {
        font-size: 1.08rem; line-height: 1.55; color: var(--text-color);
    }
    .stApp button[kind="primaryFormSubmit"], .stApp button[kind="primaryFormSubmit"] p,
    .stApp button[kind="primary"], .stApp button[kind="primary"] p { color: #fff; }
    .stApp h1 { font-size: 2rem; line-height: 1.2; }
    .stApp h2, .stApp h3 { font-size: 1.45rem; line-height: 1.3; }
    .stApp h4 { font-size: 1.375rem; line-height: 1.35; }
    .stApp [data-testid="stCaptionContainer"] { opacity: 1; color: var(--text-color); }
    .stApp [data-testid="stCaptionContainer"] p { font-size: 0.9rem; color: var(--text-color); }
    .stApp table { font-size: 1rem; }
    .stApp table th, .stApp table td { color: var(--text-color); }
    .stApp [data-testid="stText"] pre, .stApp [data-testid="stText"] span { white-space: pre-wrap; overflow-wrap: anywhere; word-break: normal; }
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] { overflow-wrap: anywhere; min-width: 0; }
    .stApp [data-testid="stTable"] { display: block; overflow-x: auto; max-width: 100%; min-width: 0; }
    .stApp [data-testid="stTable"] table { min-width: 46rem; }
    @media (max-width: 640px) { .stApp h1 { font-size: 1.8rem; } }
    </style>""", unsafe_allow_html=True)
    st.title("Ask PETzyme")
    st.caption("Investigate PET evidence and plan experiments. Source records are displayed transiently.")
    if "research_session" not in st.session_state:
        st.session_state["research_session"] = new_research_session()
    session = st.session_state["research_session"]
    pending = st.session_state.pop("unified_pending", None)
    view: ResearchView | None = None
    notice: str | None = None
    if pending:
        try:
            with st.status("Investigating your question…", expanded=False) as progress:
                def observe(update: ToolUpdate) -> None:
                    progress.write(f"{update['name']} · {update['status']} — {update['summary']}")

                def run_turn(question: str, active_session: ResearchSession):
                    """Flow: Receives prepared question str and ResearchSession from dispatch_research_turn(); returns a transient tool result for that dispatcher."""
                    separate_context = list(active_session.active_questions)
                    if separate_context and separate_context[-1] == question.strip():
                        separate_context.pop()
                    if active_session.cohort_origin is not None:
                        separate_context = []
                    if not pending["authorized"]:
                        return run_offline(question, on_update=observe, cohort_origin=active_session.cohort_origin,
                                           cohort_filter=active_session.cohort_filter,
                                           cohort_compare_filters=active_session.cohort_compare_filters,
                                           cohort_compare_scopes=active_session.cohort_compare_scopes,
                                           cohort_pazy_only=active_session.cohort_pazy_only,
                                           cohort_window=active_session.cohort_window, context_turns=separate_context)
                    dialogue_input: DialogueInput | None = None
                    if active_session.cohort_origin is not None:
                        dialogue_input = None
                    elif active_session.dialogue_context is not None:
                        dialogue_input = {"context": active_session.dialogue_context,
                                          "text": pending["question"], "input_kind": "ANSWER",
                                          "decision_support": "decision" in active_session.dialogue_context}
                    elif set(active_session.sources) == {"NREL", "PAZy"}:
                        try:
                            parse_clarification_context(question)
                            dialogue_input = {"context": None, "text": question,
                                              "input_kind": "ANSWER", "decision_support": True}
                        except ValueError:
                            # Keep the actual turn intact; the shared worker receives
                            # validated prior context separately.
                            pass
                    extra = {"dialogue_input": dialogue_input} if dialogue_input is not None else {}
                    if dialogue_input is not None:
                        separate_context = []
                    result = run_keychain_investigation(
                        question, authorized=True, on_update=observe, completion_tokens=50000,
                        automatic_runtime=True, **extra,
                        cohort_origin=active_session.cohort_origin,
                        cohort_filter=active_session.cohort_filter,
                        cohort_compare_filters=active_session.cohort_compare_filters,
                        cohort_compare_scopes=active_session.cohort_compare_scopes,
                        cohort_pazy_only=active_session.cohort_pazy_only,
                        cohort_fingerprint=active_session.cohort_fingerprint,
                        cohort_window=active_session.cohort_window,
                        context_turns=separate_context,
                    )
                    live_view = result['result']
                    unavailable = (live_view is None and result['status'] == 'ERROR') or bool(
                        live_view and live_view.get('synthesis', {}).get('reason_code') == 'BACKEND_FAILURE'
                        and not (live_view.get('observations') or live_view.get('multi_study')
                                 or live_view.get('discovery') or live_view.get('comparison_groups'))
                    )
                    if unavailable:
                        fallback = run_offline(question, on_update=observe, cohort_origin=active_session.cohort_origin,
                                               cohort_filter=active_session.cohort_filter,
                                               cohort_compare_filters=active_session.cohort_compare_filters,
                                               cohort_compare_scopes=active_session.cohort_compare_scopes,
                                               cohort_pazy_only=active_session.cohort_pazy_only,
                                               cohort_window=active_session.cohort_window, context_turns=separate_context)
                        if fallback['result'] is not None:
                            fallback['result']['limitations'].append('Muse was unavailable; these are deterministic local findings, not a Muse response.')
                            return fallback
                    return result

                updated, result, replayed = dispatch_research_turn(session, pending["question"],
                    lambda q: run_turn(q, session), context_runner=run_turn,
                    allow_adaptive=pending["authorized"])
                view = result["result"]
                if view is None:
                    if result["status"] == "NO_DATA" and (updated.pending_label or updated.pending_need or len(updated.turns) > len(session.turns)):
                        session = record_turn_status(updated, result["status"])
                        st.session_state["research_session"] = session
                        notice = result["message"]
                        progress.update(label="Clarification needed", state="complete")
                    else:
                        st.session_state["unified_error"] = result["message"]
                        progress.update(label="Investigation incomplete", state="error")
                else:
                    receipts = {r["source"]: r["measurements"] for r in view.get("source_coverage", [])
                                if r["status"] == "FOUND"}
                    session = record_turn_status(updated, result["status"], receipts.get("NREL"), receipts.get("PAZy"))
                    st.session_state["research_session"] = session
                    if replayed:
                        st.caption("Previous discovery was revalidated with local source tools; results were not saved.")
                    incomplete = view.get("synthesis", {}).get("status") == "INCOMPLETE"
                    progress.update(label="Investigation incomplete" if incomplete else "Investigation finished",
                                    state="error" if incomplete else "complete")
        except ValueError as error:
            st.session_state["unified_error"] = str(error)
        except Exception:
            st.session_state["unified_error"] = "Investigation stopped; no substitute evidence was generated."
    if session.turns:
        with st.expander("Research conversation", expanded=False):
            for turn in session.turns:
                st.write(f"{turn.number}. {turn.text or '[private input withheld]'}")
                if turn.result_status:
                    counts = ", ".join(f"{name}: {count}" for name, count in (("NREL", turn.nrel_count), ("PAZy", turn.pazy_count)) if count is not None)
                    st.caption("Execution: " + turn.result_status + (" · source observations retrieved: " + counts if counts else ""))
        st.button("New investigation", key="unified_reset", on_click=reset_unified_session)
    if view is not None:
        if view.get("dialogue"):
            render_dialogue_result(view, render_findings)
        else:
            render_findings(view)
    if notice:
        st.info(notice)
    elif session.pending_label:
        st.info(("Did you mean the NREL label " + session.pending_candidate + " for " + session.pending_label + "? Please answer yes or no.")
                if session.pending_candidate else "Please provide an accession, publication DOI, exact name or sequence for " + session.pending_label + ". No identity was substituted.")
    elif session.pending_need:
        st.info("Please choose one of the named enzymes for the single-enzyme advisor; no winner is assumed."
                if session.pending_need == "FOCUS" else
                "For a conditional two-enzyme experiment, please state the available temperatures and maximum pH.")
    if st.session_state.get("unified_error"):
        st.error(st.session_state["unified_error"])
    with st.form("ask_zymecycle", clear_on_submit=False):
        st.text_area("Research question", key="unified_question", max_chars=4000)
        st.form_submit_button("Ask PETzyme", on_click=queue_unified_turn, type="primary")
    st.caption("Muse is used when the secure credential is available; otherwise local tools run without paid inference. Results are not saved as history.")


if __name__ == "__main__":
    main()
