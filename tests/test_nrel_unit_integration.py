"""Record-bound unit inference and transient cross-source integration; no network."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_researcher_workflow import workbook_fixture

from petzyme import nrel_units
from petzyme.enzyme_discovery import discover_pet_enzymes
from petzyme.observations import load_nrel, parse_nrel
from petzyme.research_advisor import advise_next_experiment
from petzyme.ui_adapter import run_offline
from petzyme.ui_experiments import combined_experimental_rows

ROOT = Path(__file__).resolve().parents[1]
A = {"temperature_c": 40, "ph": 6.5, "pet_form": "CRYSTALLINE_POWDER"}
B = {**A, "temperature_c": 60}


def original_observations():
    return parse_nrel(load_nrel()["result"])["result"]["observations"]


def test_real_pinned_certificate_counts_and_historical_audit_are_separate():
    manifest = json.loads((ROOT / "NREL_SOURCE_MANIFEST.json").read_text())
    cert = manifest["working_unit_interpretation"]
    content = (ROOT / "data/nrel/label_data.csv").read_bytes()
    assert cert["csv_sha256"] == hashlib.sha256(content).hexdigest()
    assert cert["workbook_sha256"] == nrel_units.WORKBOOK_SHA256
    assert cert["counts"] == {
        "observations": 1570,
        "supported": 1570,
        "failed": 0,
        "direct_V_matches": 1067,
        "rounded_V_matches": 1510,
        "rounding_sensitive": 60,
        "stored_formulas": 0,
    }
    assert len(cert["supported_cells"]) == 1570 and not cert["failed_cells"]
    assert len(cert["lcc_rounding_diagnostics"]) == 4
    assert all(
        c["original"] == str(c["original"]) and "ALTERNATE_ROUNDING_DISAGREES" in c["status"]
        for c in cert["lcc_rounding_diagnostics"]
    )
    assert "CSV_UNIT_ASSIGNMENT_UNVERIFIED" in manifest["unit_reconciliation"]["status"]


def test_originals_remain_measured_and_inferred_conversion_is_separate_computation():
    raw = original_observations()
    observations = [
        {**o, "working_units": u} for o, u in zip(raw, nrel_units.standardized_nrel_values(raw))
    ]
    assert all("working_units" not in o for o in raw)
    assert sum(o["working_units"]["status"] == "INFERRED_SUPPORTED" for o in observations) == 1570
    selected = [
        o
        for o in observations
        if o["enzyme_identifier"] == "ESM050"
        and o["ph"] == 6.5
        and o["pet_form"] == "CRYSTALLINE_POWDER"
    ]
    assert [o["original_text"] for o in selected] == ["0.08499999999999999", "0.13"]
    assert all(
        o["unit_status"] == "UNKNOWN/UNVERIFIED" and o["evidence_category"] == "MEASURED"
        for o in selected
    )
    assert [round(o["working_units"]["standardized_value"], 10) for o in selected] == [85, 130]
    assert all(
        o["working_units"]["evidence_category"] == "COMPUTED"
        and o["working_units"]["conversion_factor"] == 1000
        and o["working_units"]["source_unit"].endswith("(INFERRED)")
        for o in selected
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("csv_sha256", "not-the-source"),
        ("original_text", "999"),
        ("enzyme_identifier", "other"),
        ("ph", 2.5),
        ("temperature_c", 80),
        ("pet_form", "AMORPHOUS_FILM"),
        ("source_column", "activity_at_6.5_80_cryPow"),
        ("source_row", 9999),
    ],
)
def test_rebound_or_modified_observation_cannot_use_certificate(field, value):
    observation = deepcopy(original_observations()[0])
    observation[field] = value
    representation = nrel_units.standardized_nrel_values([observation])[0]
    assert representation["status"] == "UNRESOLVED"
    assert (
        representation["standardized_value"] is None and representation["conversion_factor"] is None
    )


@pytest.mark.parametrize("problem", ["missing", "hash", "unsupported"])
def test_certificate_failure_fails_closed_for_only_affected_cells(tmp_path, monkeypatch, problem):
    manifest = json.loads((ROOT / "NREL_SOURCE_MANIFEST.json").read_text())
    observation = original_observations()[0]
    (tmp_path / "data/nrel").mkdir(parents=True)
    (tmp_path / "data/nrel/label_data.csv").write_bytes(
        (ROOT / "data/nrel/label_data.csv").read_bytes()
    )
    if problem == "hash":
        manifest["working_unit_interpretation"]["csv_sha256"] = "bad"
    if problem == "unsupported":
        del manifest["working_unit_interpretation"]["supported_cells"][
            f"{observation['source_row']}:{observation['source_column']}"
        ]
    if problem != "missing":
        (tmp_path / "NREL_SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(nrel_units, "ROOT", tmp_path)
    representation = nrel_units.standardized_nrel_values([observation])[0]
    assert representation["status"] == "UNRESOLVED" and representation["standardized_value"] is None


def test_reconstruction_uses_source_volume_round_before_mean_and_rejects_failures(monkeypatch):
    workbook = workbook_fixture()
    monkeypatch.setattr(nrel_units, "WORKBOOK_SHA256", hashlib.sha256(workbook).hexdigest())
    rows, inspection = nrel_units.read_activity_workbook(workbook)
    second = {
        **rows[0],
        "row": 3,
        "mM_product_per_mg_enzyme": 178.15,
        "mmol_product_per_mg_enzyme": 0.089,
        "umol_product_per_mg_enzyme": 89.0,
    }
    monkeypatch.setattr(
        nrel_units, "read_activity_workbook", lambda _: ([*rows, second], inspection)
    )
    source = b",activity_at_6.5_40_cryPow\nESM050,0.08499999999999999\n"
    cert = nrel_units.build_unit_certificate(source, workbook)
    assert (
        cert["counts"]["supported"] == 1
        and cert["supported_cells"]["2:activity_at_6.5_40_cryPow"] == "2, 3"
    )
    assert cert["counts"]["direct_V_matches"] == 0
    bad = nrel_units.build_unit_certificate(source.replace(b"0.08499999999999999", b"99"), workbook)
    assert not bad["supported_cells"] and len(bad["failed_cells"]) == 1
    second["umol_product_per_mg_enzyme"] = 1
    dimensional = nrel_units.build_unit_certificate(source, workbook)
    assert not dimensional["supported_cells"]
    with pytest.raises(ValueError, match="checksum"):
        nrel_units.build_unit_certificate(source, b"invalid-workbook")


def test_workbook_row_link_is_documentary_not_sequence_or_replication_proof():
    link = nrel_units.workbook_reference_link
    assert (
        link("1378, 1379", "other", "cs5c03460_si_002.xlsx Activity sheet, rows 1378, 1379")
        == "UNVERIFIED"
    )
    exact = link(
        "1378, 1379",
        "10.1021/acscatal.5c03460",
        "cs5c03460_si_002.xlsx Activity sheet, rows 1378, 1379",
    )
    assert exact.startswith("DOCUMENTED_WORKBOOK_ROW_MATCH") and "not construct identity" in exact
    assert link(
        "1378, 1379", "10.1021/acscatal.5c03460", "cs5c03460_si_002.xlsx Activity sheet, row 1378"
    ).startswith("PARTIAL_ROW_OVERLAP")


@pytest.mark.parametrize("enzyme", ["ESM050", "LCC-ICCG", "DP043"])
def test_actual_landscapes_and_offline_entrypoint_work_for_multiple_enzymes(enzyme):
    question = (
        f"Show NREL and PAZy measurements for {enzyme} at pH 6.5 using crystalline PET powder."
    )
    result = run_offline(question)
    assert result["result"] is not None
    view = result["result"]
    assert view["multi_study"]["nrel"]["observations"] and view["model_requests"] > 0
    assert view["mode"] == "DETERMINISTIC_OFFLINE"
    rows = combined_experimental_rows(view["multi_study"])
    assert any(r.source == "NREL · inferred conversion" for r in rows)
    if enzyme == "LCC-ICCG":
        assert not any(r.source == "PAZy · reported" for r in rows)
        assert len(view["multi_study"]["units"]["lcc_rounding_diagnostics"]) == 4
    else:
        assert any(r.source == "PAZy · reported" for r in rows)
    assert not view["multi_study"]["quantitative_superiority_supported"]


def test_advisor_uses_actual_pazy_endpoints_without_proposing_them_as_novel():
    advice = advise_next_experiment("ESM050", A, B)["result"]
    proposed = [
        a
        for a in advice["plan_result"]["result"]["recommendations"]
        if a["category"] == "PROPOSED_EXPERIMENT"
    ]
    assert proposed
    for action in proposed:
        assert "not present them as novel" in action["proposed_action"]
        assert any(
            b["category"] == "MEASURED"
            and "individual product endpoints" in b["statement"]
            and b["sources"]
            for b in action["evidence_basis"]
        )
        assert action["feasibility"]["status"] == "UNKNOWN"
        assert action["prerequisites"] and action["evidence_category"] == "HYPOTHETICAL"
    assert advice["persistence"] == "TRANSIENT_PAZY_SELECTION"


def test_top_five_unchanged_no_cross_source_pooling():
    value = discover_pet_enzymes()["result"]
    assert [(e["enzyme"], e["response"]) for e in value["entries"]] == [
        ("ESM050", 50.5),
        ("ESM128", 43),
        ("TEP109", 43),
        ("ESM069", 29),
        ("ESM052", 28.5),
    ]
    assert all(len(e["measurements"]) == 2 for e in value["entries"])


def test_real_streamlit_landscape_is_transient_and_originals_are_accessible(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    app = AppTest.from_file(str(ROOT / "tests/legacy_streamlit_app.py"), default_timeout=30).run()
    question = "Show NREL and PAZy measurements for ESM050 at pH 6.5 using crystalline PET powder."
    app.text_area(key="question").set_value(question)
    app.button(key="investigate").click().run()
    assert not app.exception
    tables = [t.value for t in app.table]
    combined = next(t for t in tables if "Source" in t.columns)
    nrel = combined[combined["Source"] == "NREL · inferred conversion"]
    assert set(nrel["Response"].astype(str)) == {"85", "130"}
    assert any("NREL inferred conversion" in c.value and "INFERRED" in c.value for c in app.caption)
    assert all(value == "Not recorded" for value in nrel["Duration"])
    assert any("does not establish independent replication" in m.value for m in app.markdown)
    assert any("PAZy · reported" == source for source in combined["Source"])
    assert any("0.08499999999999999" in t.value for t in app.text)
    assert any("Original CSV declaration" in c.value for c in app.caption)
    assert "view" not in app.session_state
    assert "working_units" not in str(app.session_state)
    assert "matching_parent_records" not in str(app.session_state)
    app.run()
    assert not app.exception
    assert not any("What were the results?" in m.value for m in app.markdown)


def test_bad_unit_metadata_does_not_hide_other_reported_source_measurements():
    from typing import cast

    from petzyme.pet_evidence import investigate_pet_evidence
    from petzyme.pet_presenter import preview_pet_evidence
    from petzyme.privacy import safe_public_text
    from petzyme.ui_experiments import CombinedExperimentalPreview

    native = investigate_pet_evidence({"protein_name": "ESM050", "ph": 6.5, "form": "powder"})[
        "result"
    ]
    preview = preview_pet_evidence(native, safe_public_text)
    original = deepcopy(preview)
    representation = preview["nrel"]["observations"][0]["working_units"]
    representation.update(
        status="UNRESOLVED",
        standardized_value=None,
        conversion_factor=None,
        display_unit="UNKNOWN/UNVERIFIED",
        source_unit="UNKNOWN/UNVERIFIED",
    )
    rows = combined_experimental_rows(cast(CombinedExperimentalPreview, preview))
    assert rows[0].source == "NREL · original CSV" and rows[0].response == "0.085"
    assert rows[0].unit == "UNKNOWN/UNVERIFIED"
    assert rows[1].source == "NREL · inferred conversion" and rows[1].response == "130"
    assert preview["pazy"] == original["pazy"]
    assert [r for r in rows if r.source == "PAZy · reported"]


def test_morphology_filter_does_not_assign_amorphous_powder_to_crystalline_nrel():
    from petzyme.pet_evidence import investigate_pet_evidence

    result = investigate_pet_evidence(
        {
            "protein_name": "ESM050",
            "temperature_c": 40,
            "ph": 6.5,
            "form": "powder",
            "morphology": "amorphous",
        }
    )
    assert result["result"]["nrel"]["observations"] == []
    for record in result["result"]["pazy"]["records"]:
        assert record["context"]["material"]["morphology"] == "amorphous"


def test_dimensions_must_reconcile_in_addition_to_scaling_and_group_mean(monkeypatch):
    workbook = workbook_fixture()
    monkeypatch.setattr(nrel_units, "WORKBOOK_SHA256", hashlib.sha256(workbook).hexdigest())
    rows, inspection = nrel_units.read_activity_workbook(workbook)
    # V/W agree with each other but contradict the independently recorded U/D.
    rows[0].update(mmol_product_per_mg_enzyme=0.9, umol_product_per_mg_enzyme=900)
    monkeypatch.setattr(nrel_units, "read_activity_workbook", lambda _: (rows, inspection))
    certificate = nrel_units.build_unit_certificate(
        b",activity_at_6.5_40_cryPow\nESM050,0.08\n", workbook
    )
    assert not certificate["supported_cells"] and len(certificate["failed_cells"]) == 1
