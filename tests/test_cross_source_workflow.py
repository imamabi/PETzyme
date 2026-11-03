"""Local real-source integration and synthetic coverage checks; never paid inference."""

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from streamlit.testing.v1 import AppTest

from petzyme.agent_request import interpret_request, required_tool_steps, select_tool_catalog
from petzyme.experimental_coverage import (
    MAX_COVERAGE_CONDITIONS,
    MAX_MISSING_CONDITIONS,
    summarize_experimental_coverage,
)
from petzyme.observations import load_nrel, parse_nrel
from petzyme.nrel_units import standardized_nrel_values
from petzyme.pet_evidence import investigate_pet_evidence
from petzyme.pet_presenter import preview_pet_evidence
from petzyme.pet_retrieval import load_activity_source, match_pet_assays
from petzyme.privacy import safe_public_text
from petzyme.research_advisor import advise_next_experiment
from petzyme.ui_experiments import CombinedExperimentalPreview, combined_experimental_rows

A = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
B = {**A, "temperature_c": 60}
APP = Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py"


@pytest.fixture(scope="module")
def local_sources():
    loaded = load_activity_source()
    assert loaded["status"] == "SUCCESS"
    return loaded["result"], parse_nrel(load_nrel()["result"])["result"]["observations"]


def test_real_full_enzyme_coverage_exceeds_preview_and_never_counts_replication():
    value = investigate_pet_evidence({"protein_name": "ESM050"}, limit=1)["result"]
    c = value["experimental_coverage"]
    assert (c["nrel_observations"], c["pazy_assays"], c["pazy_measurements"]) == (10, 16, 64)
    assert (c["nominal_overlap"], c["pazy_additional_conditions"], c["nrel_only_conditions"]) == (
        10,
        6,
        0,
    )
    assert c["pazy_publications"] == 1 and c["same_publication_assays"] == 16
    assert c["other_publication_assays"] == 0 and c["missing_in_both_count"] == 0
    assert len(c["conditions"]) == 16 and value["pazy"]["returned"] == 1
    assert value["identity"]["status"] == "UNVERIFIED"
    assert not value["quantitative_superiority_supported"] and not value["causal_claim_supported"]


@pytest.mark.parametrize("enzyme", ["DP003", "DP043"])
def test_other_real_curated_enzymes_have_data_driven_coverage(enzyme, local_sources):
    snapshot, observations = local_sources
    refs = match_pet_assays(snapshot, {"protein_name": enzyme})
    selected = [o for o in observations if o["enzyme_identifier"] == enzyme]
    coverage = investigate_pet_evidence({"protein_name": enzyme})["result"]["experimental_coverage"]
    assert coverage["enzyme"] == enzyme
    assert coverage["nrel_observations"] == len(selected) > 0
    assert coverage["pazy_assays"] == len(refs) > 0
    assert coverage["pazy_measurements"] == sum(len(r["assay"]["measurements"]) for r in refs)


def test_pazy_only_nominal_conditions_are_not_proposed_as_novel():
    a = {**A, "ph": 5.5, "pet_form": "AMORPHOUS_FILM"}
    b = {**a, "temperature_c": 60}
    value = advise_next_experiment("ESM050", a, b)["result"]
    assert all(not c["observations"] for c in value["condition_result"]["result"]["conditions"])
    proposals = [
        p
        for p in value["plan_result"]["result"]["recommendations"]
        if p["category"] == "PROPOSED_EXPERIMENT"
    ]
    assert len(proposals) == 2
    for action in proposals:
        assert "not present them as novel" in action["proposed_action"]
        assert "verification/replication" in action["proposed_action"]
        assert action["prerequisites"] and action["controlled_variables"]
        assert action["feasibility"]["status"] == "UNKNOWN"
        assert any("PAZy documents" in basis["statement"] for basis in action["evidence_basis"])
        assert action["evidence_category"] == "HYPOTHETICAL"
    assert not value["condition_result"]["result"]["direct_comparison_supported"]


def test_requested_missing_condition_is_not_predicted_or_filled():
    value = advise_next_experiment("ESM050", A, {**B, "temperature_c": 50})["result"]
    assert value["condition_result"]["result"]["conditions"][1]["observations"] == []
    assert value["evidence_results"][1]["result"]["pazy"]["total_matches"] == 0
    proposals = value["plan_result"]["result"]["recommendations"]
    target = next(p for p in proposals if p["code"] == "MEASURE_EXACT_CONDITION")
    assert target["target_conditions"][0]["temperature_c"] == 50
    assert target["feasibility"]["status"] == "UNKNOWN"


def test_missing_grid_is_scoped_and_bounded(local_sources):
    _, observations = local_sources
    original = observations[0]
    synthetic = [{**original, "temperature_c": t, "ph": p} for t, p in ((40, 5), (60, 5), (40, 6))]
    c = summarize_experimental_coverage("Synthetic", synthetic, [])
    assert c["missing_in_both_count"] == 1
    assert c["missing_in_both"][0]["temperature_c"] == 60
    assert c["missing_in_both"][0]["ph"] == 6
    many = [{**original, "temperature_c": t, "ph": float(t % 14)} for t in range(100)]
    c = summarize_experimental_coverage("Synthetic", many, [])
    assert len(c["conditions"]) == MAX_COVERAGE_CONDITIONS
    assert c["conditions_omitted"] == 100 - MAX_COVERAGE_CONDITIONS
    assert len(c["missing_in_both"]) == MAX_MISSING_CONDITIONS
    assert c["missing_omitted"] == c["missing_in_both_count"] - MAX_MISSING_CONDITIONS
    assert "not the whole literature" in c["basis"]
    json.dumps(c, allow_nan=False)


def test_morphology_identity_and_publications_stay_separate(local_sources):
    snapshot, observations = local_sources
    ref = deepcopy(
        match_pet_assays(
            snapshot, {"protein_name": "ESM050", "temperature_c": 40, "ph": 6.5, "form": "powder"}
        )[0]
    )
    ref["assay"]["material"]["morphology"] = "unknown"
    ref["publication_doi"] = "10.example/synthetic"
    selected = [
        o
        for o in observations
        if o["enzyme_identifier"] == "ESM050" and all(o[k] == v for k, v in A.items())
    ]
    c = summarize_experimental_coverage("ESM050", selected, [ref])
    assert c["nominal_overlap"] == 0 and c["other_publication_assays"] == 1
    assert c["same_publication_assays"] == 0
    assert c["nrel_only_conditions"] == c["pazy_additional_conditions"] == 1
    assert "unverified" in c["basis"]


def test_combined_rows_preserve_individual_values_units_and_source_refs():
    result = investigate_pet_evidence({"protein_name": "ESM050", "ph": 6.5, "form": "powder"})
    preview = preview_pet_evidence(result["result"], safe_public_text)
    rows = combined_experimental_rows(cast(CombinedExperimentalPreview, preview))
    nrel = [r for r in rows if r.source.startswith("NREL")]
    assert [r.original_response for r in nrel] == ["0.08499999999999999", "0.13"]
    assert [r.response for r in nrel] == ["85", "130"]
    assert all(r.original_unit == "UNKNOWN/UNVERIFIED" and r.duration == "Not recorded" for r in nrel)
    assert all(r.conversion["status"] == "INFERRED_SUPPORTED" and r.unit == "µmol aromatic products/mg enzyme" for r in nrel)
    assert all("Inferred from source reconciliation" in r.definition for r in nrel)
    assert all("original CSV declaration unverified" in r.definition for r in nrel)
    pazy = [r for r in rows if r.source == "PAZy · reported"]
    originals = [m["record"] for c in preview["pazy"]["records"] for m in c["measurements"]]
    assert len(pazy) == len(originals) == 6
    assert [r.original_response for r in pazy] == [str(m["value"]) for m in originals]
    assert [r.unit for r in pazy] == [m["unit"] for m in originals]
    assert all(r.duration == "48 h" and "10.1021/acscatal.5c03460" in r.reference for r in pazy)
    assert all("/measurements/" in r.reference for r in pazy)
    assert "UNRESOLVED" in result["result"]["units"]["csv_units"]
    assert result["result"]["units"]["record_examples"]
    assert all(isinstance(e["workbook_rows"], str) for e in preview["units"]["record_examples"])


def test_rounding_sensitive_cells_surface_manifest_diagnostics_without_blocking():
    observations = parse_nrel(load_nrel()["result"])["result"]["observations"]
    sensitive = [{**o, 'working_units': u} for o, u in zip(observations, standardized_nrel_values(observations))
                 if u['rounding_sensitive']]
    assert sensitive
    rows = combined_experimental_rows(
        cast(CombinedExperimentalPreview, {"nrel": {"observations": sensitive[:2]}, "pazy": {"records": []}})
    )
    assert all(r.source == "NREL · inferred conversion" for r in rows)
    assert all("alternate worksheet rounding disagrees" in r.definition for r in rows)


def test_grounded_source_coverage_and_unit_examples_preserve_privacy():
    from test_research_advisor_integration import AdviceTransport, investigate

    from petzyme.agent_reporting import format_research_report

    scenario = AdviceTransport()
    result = investigate(scenario)["result"]
    assert result["synthesis"]["status"] == "VALIDATED_FACT_SELECTION"
    assert result["validation"]["grounded"] and not result["validation"]["complete"]
    assert result["validation"]["selection_counts"]["omitted"] == 0
    assert any('"selected_enzyme"' in f["text"] for f in result["facts"])
    assert all(f["ephemeral"] for f in result["facts"])
    serialized = json.dumps(scenario.requests)
    for forbidden in (
        "/Users/",
        "amino_acid_sequence",
        "sequence_sha256",
        "MODEL_API_KEY",
        '"embedding"',
        '"vector"',
        '"rows"',
    ):
        assert forbidden not in serialized
    saved = format_research_report(result)
    assert "PAZy publication-derived assay:" not in saved
    assert '"selected_enzyme"' not in saved


def test_combined_overview_never_claims_no_measurements_with_source_rows():
    from petzyme.ui_adapter import EXAMPLES, run_offline
    from petzyme.ui_findings import result_overview

    view = run_offline(EXAMPLES["Multi-study comparison"])["result"]
    summary = result_overview(view)
    assert summary.label == "RECORDED EVIDENCE · PARTIAL"
    assert any("unverified" in s.lower() and "INFERRED" in s for s in summary.uncertainties)
    assert any("not independent replication" in s for s in summary.uncertainties)
    advice = run_offline(EXAMPLES["Integrated experiment advisor"])["result"]
    steps = result_overview(advice).steps
    assert "verification/replication" in steps[-1].text
    assert "40°C" in steps[-1].text and "60°C" in steps[-1].text
    assert "temperature_c" not in steps[-1].text


def test_new_coverage_projection_screens_sequence_like_labels():
    value = investigate_pet_evidence({"protein_name": "A" * 50})["result"]
    preview = preview_pet_evidence(value, safe_public_text)
    assert "A" * 50 not in json.dumps(preview)
    assert preview["experimental_coverage"]["enzyme"] == "[sequence-like text withheld]"


def test_exploratory_catalog_requires_cross_source_composition():
    # Supersedes the without-forcing expectation: a 2026-10-31 live run with the
    # composition exposed-but-optional completed NREL-only (load/parse/filter/assess)
    # and never selected it. Broad planning therefore requires the local composition.
    q = "What conditions should I test next for ESM050?"
    request = interpret_request(q)
    catalog = {t["function"]["name"] for t in select_tool_catalog(request)}
    assert "investigate_pet_evidence" in catalog and "load_nrel" in catalog
    assert "investigate_pet_evidence" in required_tool_steps(request)
    explicit = interpret_request(q + " Use NREL and PAZy evidence.")
    assert "investigate_pet_evidence" in required_tool_steps(explicit)
    assert {"investigate_pet_evidence", "advise_next_experiment"} <= {t["function"]["name"] for t in select_tool_catalog(explicit)}
    nrel_only = interpret_request(q + " Use NREL only.")
    assert "investigate_pet_evidence" not in {
        t["function"]["name"] for t in select_tool_catalog(nrel_only)
    }


@pytest.mark.parametrize("index", [6, 7])
def test_real_ui_combined_findings_transient_reset_and_no_inference(index, monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Forbidden inference, credential or external API access")

    monkeypatch.setattr("petzyme.meta.MetaModelClient.complete", blocked)
    monkeypatch.setattr("petzyme.ui_live.subprocess.Popen", blocked)
    monkeypatch.setattr("petzyme.pazy.PazyClient._get", blocked, raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key=f"example_{index}").click().run()
    at.button(key="investigate").click().run()
    assert not at.exception
    headings = [m.value for m in at.markdown if m.value.startswith("####")]
    for title in (
        "What has been tested?",
        "What were the results? · MEASURED / COMPUTED",
        "What is interesting or missing? · COMPUTED",
    ):
        assert "#### " + title in headings
    assert at.table and not at.dataframe and not at.json and not at.get("download_button")
    assert any(e.label == "Exact responses, definitions and assay references" for e in at.expander)
    assert any("PAZy adds 6" in c.value for c in at.caption)
    assert "view" not in at.session_state
    assert "matching_parent_records" not in str(at.session_state)
    at.run()
    assert not at.exception and not at.table and not at.text


def test_unavailable_pazy_does_not_establish_novelty(monkeypatch):
    from petzyme.results import outcome

    monkeypatch.setattr(
        "petzyme.research_advisor.investigate_pet_evidence",
        lambda *a, **k: outcome("ERROR", None, [], [], "COMPUTED"),
    )
    result = advise_next_experiment("ESM050", A, B)
    assert result["status"] == "PARTIAL"
    actions = result["result"]["plan_result"]["result"]["recommendations"]
    assert all(
        any("novelty is not established" in note for note in a["limitations"])
        for a in actions
        if a["category"] == "PROPOSED_EXPERIMENT"
    )
