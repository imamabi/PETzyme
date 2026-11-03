"""Presentation acceptance over real local evidence; no paid inference or new scientific logic."""

from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest
from test_researcher_workflow import QUESTION, PlanningBackend
from test_scientific_scope import ENDPOINT, KINETIC, POLYPROPYLENE, AllPetFacts

from zymecycle.runtime_config import RuntimeLimits
from zymecycle.ui_adapter import EXAMPLES, investigate_with_backend, run_offline
from zymecycle.ui_findings import evidence_snapshot, result_overview


def case_view(name):
    """Build actual local evidence using explicit mock selection, never a claimed live model evaluation."""
    questions = {
        "Planning": QUESTION,
        "PET comparison": ENDPOINT,
        "Polypropylene": POLYPROPYLENE,
        "Kinetics": KINETIC,
    }
    if name in questions:
        return investigate_with_backend(
            questions[name], PlanningBackend() if name == "Planning" else AllPetFacts()
        )["result"]
    if name == "Incomplete":
        return investigate_with_backend(
            ENDPOINT, AllPetFacts(), runtime_limits=RuntimeLimits(turn_models=3)
        )["result"]
    return run_offline(EXAMPLES[name])["result"]


@pytest.mark.parametrize(
    "name", ["Planning", "PET comparison", "Polypropylene", "Kinetics", "Incomplete"]
)
def test_summary_and_snapshot_preserve_source_results_without_mutation(name):
    view = case_view(name)
    original = deepcopy(view)
    summary = result_overview(view)
    groups = evidence_snapshot(view)
    assert view == original
    assert summary.answer and 2 <= len(summary.steps) <= 5
    assert len(summary.uncertainties) <= 4
    assert all(len(g.rows) <= 3 and g.total >= len(g.rows) for g in groups)
    assert all("INFERRED conversion; COMPUTED" in g.units for g in groups)
    assert not any(
        s in summary.answer
        for s in (
            "fact-reference contract",
            "publishable structured evidence",
            "grounded=True",
            "request budget exhausted",
        )
    )


def test_planning_starting_points_are_source_conditions_not_ranked_or_fabricated():
    view = case_view("Planning")
    summary = result_overview(view)
    assert summary.label == "EXPLORATORY PLANNING · PARTIAL"
    assert "no proven optimum" in summary.answer
    assert len(summary.starting_points) == 2
    for point, o in zip(summary.starting_points, view["observations"]):
        assert (
            o["enzyme_identifier"] in point
            and f"{o['temperature_c']:g}°C" in point
            and f"pH {o['ph']:g}" in point
        )
    assert all(s.label in ("CLARIFICATION", "VERIFICATION") for s in summary.steps)
    assert not view["plan"]  # The UI does not manufacture scientific proposals.


@pytest.mark.parametrize(
    "name,phrase",
    [("Polypropylene", "polypropylene degradation"), ("Kinetics", "do not establish kcat/KM")],
)
def test_scope_rejections_stay_blocking_and_background_only(name, phrase):
    view = case_view(name)
    summary = result_overview(view)
    assert summary.blocking and summary.background_only
    assert phrase in summary.answer and "PARTIAL" in summary.label
    assert view["synthesis"]["status"] == "INCOMPLETE"
    assert any("background only" in s.text for s in summary.steps)


def test_incomplete_live_wording_is_plain_and_does_not_claim_completed_muse_reasoning():
    view = case_view("Incomplete")
    view["mode"] = "LIVE_META"  # Synthetic display state only; no real provider call.
    summary = result_overview(view)
    assert (
        summary.answer
        == "Muse did not complete a final narrative answer in this run. The measured evidence below is still available."
    )
    assert not summary.blocking and view["observations"]
    assert "INCOMPLETE" in summary.label


@pytest.mark.parametrize(
    "name", ["Planning", "PET comparison", "Polypropylene", "Kinetics", "Incomplete"]
)
def test_rendered_hierarchy_details_and_exact_values(name):
    source = """
from test_ui_clarity import case_view
from streamlit_app import render_findings
render_findings(case_view(NAME))
""".replace("NAME", repr(name))
    app = AppTest.from_string(
        'import sys\nsys.path.insert(0,"tests")\n' + source, default_timeout=30
    ).run()
    assert not app.exception
    outer = next(e for e in app.expander if e.label == "Details · complete evidence, sources and execution")
    assert outer.label == "Details · complete evidence, sources and execution"
    assert outer.proto.expanded is False
    headings = [m.value for m in app.markdown]
    next_index = headings.index("#### Suggested next step")
    uncertainty_index = headings.index("#### What remains uncertain")
    if "#### What were the results? · MEASURED / COMPUTED" in headings:
        evidence_index = headings.index("#### What were the results? · MEASURED / COMPUTED")
        assert evidence_index < next_index < uncertainty_index
    else:
        evidence_index = headings.index("#### Evidence snapshot · MEASURED")
        assert next_index < uncertainty_index < evidence_index
    assert app.table  # Compact static snapshots have no export controls.
    text = " ".join(m.value for m in app.markdown)
    assert ("0.08499999999999999" in text or any("0.08499999999999999" in j.value for j in app.json)
            or any("ORIGINAL 0.08499999999999999" in t.value for t in app.text))
    if name in ("Polypropylene", "Kinetics"):
        assert any("requested comparison is unsupported" in e.value for e in app.error)
    if name == "Planning":
        assert "Observed starting points" in text and "not ranked recommendations" in " ".join(
            c.value for c in app.caption
        )


@pytest.mark.parametrize(
    "name",
    [
        "Temperature evidence",
        "Missing conditions",
        "Next experiment",
        "Integrated experiment advisor",
    ],
)
def test_existing_offline_flows_keep_all_hypothetical_warnings_and_originals(name):
    view = case_view(name)
    summary = result_overview(view)
    if view["plan"]:
        assert all(s.label.startswith("HYPOTHETICAL") for s in summary.steps)
        assert summary.steps[0].text == view["plan"]["recommendations"][0]["proposed_action"]
    assert view["mode"] == "DETERMINISTIC_OFFLINE"
    assert "no live Muse reasoning" in summary.attribution


def test_snapshot_keeps_forms_separate_and_never_sorts_by_response():
    view = case_view("Planning")
    groups = evidence_snapshot(view)
    assert {g.name for g in groups} == {"Crystalline Powder", "Amorphous Film"}
    for g in groups:
        form = g.name.upper().replace(" ", "_")
        originals = [o for o in view["observations"] if o["pet_form"] == form]
        assert [r["ph"] for r in g.rows] == [f"{o['ph']:g}" for o in originals[:3]]
        assert all(set(r) == {"enzyme", "temperature", "ph", "response"} for r in g.rows)


def test_similarity_only_view_never_claims_measured_pet_observations():
    view = case_view("PET comparison")
    view["observations"] = []
    view["comparison"] = None
    view["similarity"] = {
        "neighbors": [{"identifier": "SYNTHETIC_NEIGHBOR", "cosine_similarity": 0.5}],
        "query_identifier": "SYNTHETIC_QUERY",
        "coverage": {"searched_candidates": 1},
    }
    view["scientific_scope"]["background_only"] = True
    summary = result_overview(view)
    assert "computed similarity" in summary.answer
    assert not any("Available PET observations" in text for text in summary.uncertainties)
    assert any("does not establish PET activity" in text for text in summary.uncertainties)
    assert summary.steps[0].label == "COMPUTED EVIDENCE"


def test_result_focus_is_submission_only_and_does_not_repeat_scientific_execution(monkeypatch):
    from pathlib import Path

    from zymecycle import ui_adapter

    calls = []
    original = ui_adapter.run_offline

    def run(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ui_adapter, "run_offline", run)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py", default_timeout=30
    ).run()
    assert not app.get("iframe")
    app.button(key="investigate").click().run()
    assert not app.exception and len(calls) == 1 and len(app.get("iframe")) == 1
    app.run()
    assert len(calls) == 1 and not app.get("iframe")


def test_missing_condition_summary_does_not_label_absence_as_measured_evidence():
    view = case_view("Missing conditions")
    summary = result_overview(view)
    assert not view["observations"]
    assert summary.label == "NO EXACT MEASUREMENTS · PARTIAL"
    assert "Missing is not zero" in summary.answer
