"""Readable conditional experiment presentation; validation remains in scientific units."""

from __future__ import annotations

import streamlit as st

from .decision_types import Control, Criterion, Design, ExperimentalDecision, ExperimentOption

_DESIGN_LABELS: dict[Design, str] = {
    "MATCHED_ENZYME_COMPARISON": "Matched enzyme comparison",
    "BALANCED_ENZYME_TEMPERATURE": "Enzyme-by-temperature comparison",
    "MEASUREMENT_QUALIFICATION": "Measurement qualification",
}
_REASON_LABELS: dict[Criterion, str] = {
    "EQUIPMENT_FEASIBILITY": "the alternative conflicts with the current equipment constraints",
    "TEMPERATURE_COVERAGE": "it compares both enzymes across more temperatures",
    "FOCUSED_DESIGN": "it uses fewer enzyme–temperature combinations (not a cost estimate)",
    "DIRECT_ENZYME_DECISION": "it directly compares both enzymes rather than qualifying a single-enzyme measurement",
    "MEASUREMENT_READINESS": "it addresses recorded measurement-readiness gaps before comparison",
}


def experiment_summary(option: ExperimentOption) -> str:
    """Flow: Receives validated ExperimentOption from render_experimental_decision(); returns readable str for its conditional recommendation, without adding assay settings or outcomes."""
    names = " and ".join(option["enzymes"])
    temperatures = " and ".join(f"{t:g}°C" for t in option["temperatures_c"])
    form = {"CRYSTALLINE_POWDER": "crystalline PET powder", "AMORPHOUS_FILM": "amorphous PET film"}[
        option["pet_form"]
    ]
    return f"{_DESIGN_LABELS[option['design']]}: {names} at {temperatures}, proposed pH {option['ph']:g}, using {form}."


# Presentation taxonomy only: the existing six-item validation contract is unchanged.
_REQUIREMENT_GROUPS: tuple[tuple[str, tuple[Control, ...]], ...] = (
    ("Experimental controls", ("SUBSTRATE_BLANK", "REFERENCE_CONTROL")),
    ("Replication design", ("INDEPENDENT_REPLICATION",)),
    ("Analytical validation", ("MEASUREMENT_CALIBRATION", "DETECTION_LIMIT")),
    ("Uncertainty/statistical reporting", ("UNCERTAINTY",)),
)


def render_experimental_decision(decision: ExperimentalDecision, mode: str) -> None:
    """Flow: Receives validated ExperimentalDecision from render_dialogue_result(); returns None after transient display of Muse choices and deterministic scientific checks."""
    by_id = {a["proposal"]["option_id"]: a for a in decision["alternatives"]}
    selected = by_id[decision["selected_id"]]
    st.success("Conditional next experiment: " + decision["selected_id"] + " · HYPOTHETICAL")
    st.write(experiment_summary(selected["proposal"]))
    st.markdown("#### What to measure")
    for measurement in selected["candidate"]["required_measurements"]:
        st.write("• " + measurement)
    st.markdown("#### Required next action")
    if selected["candidate"]["prerequisites"]:
        st.write(selected["candidate"]["prerequisites"][0])
        st.caption(
            "Complete the remaining source and laboratory prerequisites in the option details before measurement."
        )
    if selected["resource_feasibility"]["status"] == "UNKNOWN":
        st.warning(
            "Overall laboratory/resource feasibility is UNKNOWN. This recommendation remains conditional: "
            "verify substrate availability, assay calibration and required resources before running it. "
            "Equipment compatibility alone does not establish feasibility."
        )
    st.caption(
        "Muse chose the designs, factor levels and priority; Python verified their structured meaning and renders canonical descriptions. This is not free-form validated Muse prose. Scientific decision: PARTIAL."
        if mode == "LIVE_META"
        else "Mocked model choices with real local tools; not live Muse acceptance. Scientific decision: PARTIAL."
    )
    st.markdown("#### Why this experiment comes first")
    for reason in decision["rationale"]:
        st.write(
            "Chosen over current "
            + reason["over"]
            + " because "
            + _REASON_LABELS[reason["criterion"]]
            + "."
        )
    if decision["previous_option_status"]:
        st.write("Reassessment of previous hypothetical options")
        for identifier, status in decision["previous_option_status"].items():
            label = (
                "Requires a temperature no longer available — excluded from the previous plan"
                if status == "BLOCKED_BY_UPDATED_TEMPERATURES"
                else "Still within temperature bounds; laboratory prerequisites remain"
            )
            st.text("Previous " + identifier + ": " + label)
        st.caption(
            "Previous IDs refer to the prior turn. Current IDs below are local to this turn; the same ID can describe a revised design."
        )
    st.text(
        "Equipment-compatible conditional options in this turn: "
        + ", ".join(decision["feasible_option_ids"])
    )
    if decision["excluded_ids"]:
        st.warning(
            "Removed from feasible recommendations under current constraints: "
            + ", ".join(decision["excluded_ids"])
        )
    for identifier in decision["ranking"]:
        a = by_id[identifier]
        with st.expander(
            "Current " + identifier + " · " + _DESIGN_LABELS[a["proposal"]["design"]],
            expanded=identifier == decision["selected_id"],
        ):
            if identifier in decision["excluded_ids"]:
                st.warning("Excluded consideration, not a recommended experiment.")
            st.write(experiment_summary(a["proposal"]))
            st.write("Objective: " + a["candidate"]["objective"])
            st.text(
                "COMPUTED equipment assessment: "
                + (
                    "Within the declared temperature/pH bounds"
                    if a["equipment_status"] == "WITHIN_DECLARED_BOUNDS"
                    else "Blocked by current constraints"
                )
            )
            st.text(
                "Resources/overall laboratory feasibility: " + a["resource_feasibility"]["status"]
            )
            for blocker in a["equipment_blockers"] + a["resource_feasibility"]["blockers"]:
                st.warning(blocker)
            with st.expander("Resources, construct and protocol verification"):
                for requirement in a["resource_feasibility"]["unknown_requirements"]:
                    st.write(
                        "• "
                        + requirement.replace("Availability of ", "Availability: ")
                        .replace("_", " ")
                        .lower()
                    )
                for name, status in a["identity_status"].items():
                    st.write(name + ": source label only; experimental construct unverified.")
                st.write(
                    "Still to verify: "
                    + ", ".join(k.replace("_", " ") for k in a["metadata_status"])
                )
                st.caption(
                    "Proposed conditions remain hypothetical. Their source-coverage status is available in technical details."
                )
            st.write(a["hypothesis"])
            st.write("Information sought: " + a["candidate"]["expected_information"])
            st.text("Controlled: " + "; ".join(a["candidate"]["controlled_variables"]))
            st.text("Varied: " + "; ".join(a["candidate"]["varied_variables"]))
            for label, codes in _REQUIREMENT_GROUPS:
                requirements = [
                    code.replace("_", " ").lower()
                    for code in codes
                    if code in a["proposal"]["controls"]
                ]
                st.text(label + ": " + ", ".join(requirements))
            for measurement in a["candidate"]["required_measurements"]:
                st.write(measurement)
            for prerequisite in a["candidate"]["prerequisites"]:
                st.caption("Prerequisite: " + prerequisite)
            for implication in a["outcome_implications"]:
                st.write(implication)
    with st.expander("Technical decision details · original identifiers"):
        for explanation in decision["rationale_explanations"]:
            st.text(explanation)
        st.text("Previous option status: " + str(decision["previous_option_status"]))
        for a in decision["alternatives"]:
            st.text(a["candidate"]["proposed_action"])
            st.text("Equipment code: " + a["equipment_status"])
            st.text("Identity: " + str(a["identity_status"]))
            st.text("UNKNOWN metadata: " + str(a["metadata_status"]))
            st.text("Proposed pH basis: " + a["proposed_ph_origin"])
            for requirement in a["resource_feasibility"]["unknown_requirements"]:
                st.text("Unverified resource requirement: " + requirement)
    with st.expander("MEASURED original NREL values and provenance"):
        for fact in decision["evidence_facts"]:
            st.text("[" + fact["id"] + "] " + fact["text"])
            for source_index, source in enumerate(fact["sources"]):
                st.link_button(
                    source["name"] + " source",
                    source["url"],
                    key="decision_source_" + fact["id"] + "_" + str(source_index),
                )
    with st.expander("COMPUTED evidence quality and comparability"):
        st.text(decision["comparability"]["category"])
        for issue in decision["evidence_assessment"]["issues"]:
            st.text(issue["code"] + ": " + issue["explanation"])
    for limitation in decision["limitations"]:
        st.caption(limitation)
