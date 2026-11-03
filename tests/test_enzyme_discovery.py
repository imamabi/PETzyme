"""Synthetic protocol tests and independent real-source discovery acceptance; no paid API."""

import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from zymecycle.agent import run_investigation
from zymecycle.agent_request import interpret_request, required_tool_steps, select_tool_catalog
from zymecycle.enzyme_discovery import UNIT, discover_pet_enzymes, rank_discovery_group
from zymecycle.meta import MetaModelClient
from zymecycle.pet_retrieval import load_activity_source, match_pet_assays
from zymecycle.privacy import outbound_safe, tool_privacy
from zymecycle.results import outcome
from zymecycle.tools import invoke_tool
from zymecycle.ui_adapter import run_offline

QUESTION = "What are the 5 enzymes with the best response for PET breakdown?"
SECOND = "Show the top five PET hydrolases on amorphous film at 60C, pH 6.5."
APP = Path(__file__).resolve().parents[1] / "tests/legacy_streamlit_app.py"


def synthetic_snapshot():
    source = {
        "name": "Synthetic PAZy fixture",
        "doi": None,
        "url": "https://example.invalid/",
        "locator": "synthetic",
        "attribution": "Synthetic only",
        "retrieved_at_utc": None,
    }
    records = []
    for index in range(6):
        measurements = [
            {
                "metric_type": "specific_activity",
                "unit": UNIT,
                "analyte": "aromatic products",
                "value": index + replicate,
                "qualifier": None,
                "replicate_n": 2,
                "value_is_digitized": False,
                "reported_result": f"Biological duplicate {replicate} of 2: synthetic fixture",
            }
            for replicate in (1, 2)
        ]
        assay = {
            "substrate": "PET",
            "is_representative": False,
            "conditions": {
                "temperature_c": 40,
                "ph": 6.5,
                "duration_value": 48,
                "duration_unit": "h",
                "buffer": "synthetic buffer",
                "enzyme_preparation": "purified",
                "enzyme_amount_value": 5,
                "enzyme_amount_unit": "µg",
                "substrate_amount_value": 2,
                "substrate_amount_unit": "mg",
                "reaction_volume_value": 0.5,
                "reaction_volume_unit": "mL",
            },
            "material": {
                "reported_name": "synthetic PET",
                "form": "powder",
                "morphology": "crystalline",
                "crystallinity_percent": 30,
            },
            "detection_methods": [
                {"method": "synthetic quantitative method", "role": "quantitative"}
            ],
            "measurements": measurements,
            "evidence": {"source": "synthetic", "location": f"synthetic rows {index}"},
        }
        records.append(
            {
                "protein_id": index + 1,
                "protein_name": f"Synthetic{index}",
                "synonyms": [],
                "literature_doi": "10.example/synthetic",
                "assays": [assay],
            }
        )
    return {
        "records": records,
        "schema_version": "4.2",
        "sha256": "a" * 64,
        "source": source,
        "counts": {"parents": 6, "assays": 6, "measurements": 12, "representative_assays": 0},
    }


def test_exact_question_routes_to_independent_scientific_tool():
    request = interpret_request(QUESTION)
    assert request["objectives"] == ["enzyme_discovery"]
    assert required_tool_steps(request) == {"discover_pet_enzymes"}
    assert {t["function"]["name"] for t in select_tool_catalog(request)} == {"discover_pet_enzymes", "investigate_pet_evidence", "advise_next_experiment"}
    assert tool_privacy("discover_pet_enzymes").persistence == "FORBIDDEN_PAZY_DERIVED"
    result = invoke_tool("discover_pet_enzymes", {})
    assert result["status"] == "PARTIAL" and len(result["result"]["entries"]) == 5


@pytest.mark.parametrize("question", [QUESTION, SECOND])
def test_real_entries_independently_match_original_curated_measurements_and_units(question):
    result = run_offline(question)
    view = result["result"]
    value = view["discovery"]
    group = value["group"]
    assert (
        result["status"] == "PARTIAL" and view["model_requests"] == 0 and view["tool_attempts"] == 1
    )
    assert len(value["entries"]) == 5 and len({e["enzyme"] for e in value["entries"]}) == 5
    assert (
        group["eligible_enzymes"] >= 5 and value["omitted_enzymes"] == group["eligible_enzymes"] - 5
    )
    snapshot = load_activity_source()["result"]
    assert group["protocol"]["archive_sha256"] == snapshot["sha256"]
    assert group["protocol"]["activity_archive"] == snapshot["source"]["url"]
    independently_calculated = []
    for entry in value["entries"]:
        refs = match_pet_assays(
            snapshot,
            {
                "protein_name": entry["enzyme"],
                "temperature_c": group["temperature_c"],
                "ph": group["ph"],
                "form": "powder" if group["pet_form"] == "CRYSTALLINE_POWDER" else "film",
            },
        )
        ref = next(
            r
            for r in refs
            if f"#/{r['parent_index']}/assays/{r['assay_index']}" == entry["assay_locator"]
        )
        originals = [
            m
            for m in ref["assay"]["measurements"]
            if m.get("metric_type") == "specific_activity" and m.get("unit") == UNIT
        ]
        assert len(originals) == 2 and all(m["replicate_n"] == 2 for m in originals)
        assert [m["value"] for m in entry["measurements"]] == [m["value"] for m in originals]
        assert all(
            m["evidence_category"] == "MEASURED" and m["unit"] == UNIT
            for m in entry["measurements"]
        )
        assert math.isclose(entry["response"], mean(m["value"] for m in originals), rel_tol=1e-12)
        independently_calculated.append(entry["response"])
        assert entry["publication_doi"] == ref["publication_doi"]
        assert not value["global_ranking_supported"]
    assert independently_calculated == sorted(independently_calculated, reverse=True)
    if question == SECOND:
        assert (
            group["temperature_c"] == 60
            and group["ph"] == 6.5
            and group["pet_form"] == "AMORPHOUS_FILM"
        )
        assert all(
            not e["nrel_references"] for e in value["entries"]
        )  # PAZy evidence survives absent CSV cells.


def test_group_is_chosen_by_coverage_not_largest_activity():
    snapshot = synthetic_snapshot()
    outlier = deepcopy(snapshot["records"][0])
    outlier["protein_id"] = 99
    outlier["protein_name"] = "OtherStudy"
    outlier["literature_doi"] = "10.example/other"
    for m in outlier["assays"][0]["measurements"]:
        m["value"] = 999999
    snapshot["records"].append(outlier)
    value = rank_discovery_group(snapshot, None)["result"]
    assert value["group"]["eligible_enzymes"] == 6
    assert value["entries"][0]["enzyme"] == "Synthetic5" and value["entries"][0]["response"] == 6.5
    assert all(e["enzyme"] != "OtherStudy" for e in value["entries"])


@pytest.mark.parametrize(
    "mutation", ["unit", "buffer", "duration", "detection", "morphology", "loading"]
)
def test_incompatible_measurements_are_never_pooled(mutation):
    snapshot = synthetic_snapshot()
    assay = snapshot["records"][0]["assays"][0]
    if mutation == "unit":
        assay["measurements"][0]["unit"] = "%"
    elif mutation == "buffer":
        assay["conditions"]["buffer"] = "different buffer"
    elif mutation == "duration":
        assay["conditions"]["duration_value"] = 24
    elif mutation == "detection":
        assay["detection_methods"][0]["method"] = "different method"
    elif mutation == "morphology":
        assay["material"]["morphology"] = "amorphous"
    else:
        assay["conditions"]["enzyme_amount_value"] = 10
    value = rank_discovery_group(snapshot, None)["result"]
    assert value["group"]["eligible_enzymes"] == 5
    assert all(e["enzyme"] != "Synthetic0" for e in value["entries"])


def test_duplicates_do_not_multiply_evidence_or_choose_best_assay():
    snapshot = synthetic_snapshot()
    snapshot["records"][0]["assays"].append(deepcopy(snapshot["records"][0]["assays"][0]))
    value = rank_discovery_group(snapshot, None)["result"]
    assert value["group"]["eligible_enzymes"] == 6
    snapshot["records"][0]["assays"][1]["evidence"]["location"] = "different experiment"
    snapshot["records"][0]["assays"][1]["measurements"][0]["value"] = 9999
    value = rank_discovery_group(snapshot, None)["result"]
    assert value["group"]["eligible_enzymes"] == 5 and all(
        e["enzyme"] != "Synthetic0" for e in value["entries"]
    )


def test_documented_alias_ambiguity_is_not_silently_merged():
    snapshot = synthetic_snapshot()
    snapshot["records"][0]["synonyms"] = [{"name": "SharedAlias"}]
    snapshot["records"][1]["synonyms"] = [{"name": "SharedAlias"}]
    value = rank_discovery_group(snapshot, None)["result"]
    assert value["group"]["eligible_enzymes"] == 4
    assert not {"Synthetic0", "Synthetic1"} & {e["enzyme"] for e in value["entries"]}


def test_nrel_units_or_unavailable_reference_never_block_valid_pazy(monkeypatch):
    monkeypatch.setattr(
        "zymecycle.enzyme_discovery.load_nrel", lambda *a: outcome("ERROR", None, [], [], None)
    )
    result = discover_pet_enzymes()
    assert result["status"] == "PARTIAL" and len(result["result"]["entries"]) == 5
    assert all(not e["nrel_references"] for e in result["result"]["entries"])
    assert any("NREL reference/audit unavailable" in x for x in result["limitations"])


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 6}, {"limit": 0}, {"limit": True}, {"ph": 99}, {"temperature_c": float("nan")}],
)
def test_invalid_inputs_fail_explicitly(kwargs):
    assert discover_pet_enzymes(**kwargs)["status"] == "ERROR"


def test_unmeasured_filter_has_no_fabricated_result():
    result = discover_pet_enzymes(temperature_c=999)
    assert result["status"] == "NO_DATA" and not result["result"]["entries"]


class DiscoveryTransport:
    def __init__(self, incomplete=False):
        self.requests = []
        self.incomplete = incomplete

    def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        replies = [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]
        if not replies:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "discovery",
                        "type": "function",
                        "function": {"name": "discover_pet_enzymes", "arguments": "{}"},
                    }
                ],
            }
            reason = "tool_calls"
        else:
            ids = replies[-1]["final_selection"]["required_fact_ids"]
            msg = {
                "role": "assistant",
                "content": json.dumps({"fact_ids": ids[:-1] if self.incomplete else ids}),
            }
            reason = "stop"
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": reason}]})


@pytest.mark.parametrize("incomplete", [False, True])
def test_production_dispatcher_canonical_grounding_and_privacy(incomplete):
    scenario = DiscoveryTransport(incomplete)
    report = run_investigation(
        QUESTION, MetaModelClient(transport=httpx.MockTransport(scenario.respond))
    )["result"]
    assert report["status"] == "PARTIAL" and len(scenario.requests) == 2
    assert report["validation"][
        "grounded"
    ]  # Selected IDs exist; completeness is checked separately.
    assert report["validation"]["selection_counts"]["omitted"] == int(incomplete)
    assert not report["validation"]["complete"]
    assert (report["synthesis"]["status"] == "VALIDATED_FACT_SELECTION") is (not incomplete)
    assert report["persistence"] == "FORBIDDEN_PAZY_DERIVED"
    assert outbound_safe(scenario.requests)
    assert "/Users/" not in json.dumps(scenario.requests)


@pytest.mark.parametrize("question", [QUESTION, SECOND])
def test_exact_browser_ui_contract_without_paid_calls_or_result_history(question, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("Paid provider, Keychain or external protein API called")

    monkeypatch.setattr("zymecycle.ui_live.run_keychain_investigation", forbidden)
    monkeypatch.setattr("zymecycle.meta.MetaModelClient.complete", forbidden)
    monkeypatch.setattr("zymecycle.pazy.PazyClient._get", forbidden, raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="question").input(question)
    at.button(key="investigate").click().run()
    assert not at.exception and len(at.table) == 1
    assert len(at.table[0].value) == 5
    assert any(question in x.value for x in at.caption)
    assert any(x.value.startswith("Run ") for x in at.caption)
    assert not at.json and not at.dataframe and not at.get("download_button")
    assert not at.get("warning")
    assert "view" not in at.session_state and "source_protein_id" not in str(at.session_state)
    at.run()
    assert not at.table and not at.exception


def test_no_ranking_on_polymer_or_kinetic_scope_mismatch():
    for q in (
        "What are the 5 best enzymes for polypropylene breakdown?",
        "What are the 5 PET enzymes with the best kcat/KM?",
    ):
        assert "enzyme_discovery" not in interpret_request(q)["objectives"]


def test_fresh_run_identifiers_are_not_stale():
    first, second = run_offline(QUESTION)["result"], run_offline(QUESTION)["result"]
    assert (
        first["run_id"] != second["run_id"]
        and first["submitted_question"] == second["submitted_question"] == QUESTION
    )


def test_live_mode_without_authorization_still_returns_local_database_results(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Local discovery attempted paid inference or Keychain access")

    monkeypatch.setattr("zymecycle.ui_live.run_keychain_investigation", blocked)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Live Meta agent").run()
    at.text_area(key="question").input(QUESTION)
    at.button(key="investigate").click().run()
    assert not at.exception and len(at.table[0].value) == 5
    assert any("no Muse inference" in c.value for c in at.caption)
    assert "view" not in at.session_state


def test_explicit_nrel_scope_preserves_existing_tool_path():
    request = interpret_request(
        "Which NREL enzymes have highest PET responses at 40C pH 6.5 in powder?"
    )
    assert "enzyme_discovery" not in request["objectives"]
    assert {"load_nrel", "summarize_condition"} <= required_tool_steps(request)


def test_scientific_discovery_never_writes_logs_or_calls_network(monkeypatch, capsys, caplog):
    import builtins
    import io

    original, original_io = builtins.open, io.open

    def read_only(opener):
        def checked(file, mode="r", *args, **kwargs):
            assert not set(mode) & set("wax+")
            return opener(file, mode, *args, **kwargs)

        return checked

    def blocked(*args, **kwargs):
        pytest.fail("Local instrument accessed the network")

    monkeypatch.setattr(builtins, "open", read_only(original))
    monkeypatch.setattr(io, "open", read_only(original_io))
    monkeypatch.setattr(httpx.Client, "request", blocked)
    assert len(discover_pet_enzymes()["result"]["entries"]) == 5
    captured = capsys.readouterr()
    assert not captured.out and not captured.err and not caplog.text


@pytest.mark.parametrize(
    "question,issue",
    [
        (
            "What are the 5 best enzymes for polypropylene breakdown?",
            "SCIENTIFIC_SCOPE_MATERIAL_MISMATCH",
        ),
        ("What are the 5 PET enzymes with the best kcat/KM?", "SCIENTIFIC_SCOPE_QUANTITY_MISMATCH"),
    ],
)
def test_authentic_discovery_fact_ids_cannot_bypass_scope(question, issue):
    from zymecycle.agent_execution import execute_tool_call
    from zymecycle.agent_reporting import validate_investigation
    from zymecycle.agent_types import AgentState

    state = AgentState("data/nrel/label_data.csv")
    call = {
        "id": "discovery",
        "type": "function",
        "function": {"name": "discover_pet_enzymes", "arguments": "{}"},
    }
    execute_tool_call(call, state, select_tool_catalog(interpret_request(QUESTION)))
    ids = [f["id"] for f in state.facts if f["kind"] == "EVIDENCE"]
    validation = validate_investigation(interpret_request(question), state, {"fact_ids": ids})
    assert validation["grounded"] and not validation["complete"]
    assert issue in validation["issues"] and validation["scientific_scope"]["background_only"]


def test_mock_orchestration_is_never_labelled_real_muse(monkeypatch):
    from zymecycle import ui_adapter

    scenario = DiscoveryTransport()

    def mocked_run(question, **kwargs):
        return ui_adapter.investigate_with_backend(
            question,
            MetaModelClient(transport=httpx.MockTransport(scenario.respond)),
            on_update=kwargs.get("on_update"),
        )

    monkeypatch.setattr(ui_adapter, "run_offline", mocked_run)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="question").input(QUESTION)
    at.button(key="investigate").click().run()
    assert not at.exception and len(at.table[0].value) == 5
    assert any("Mock/offline orchestration" in c.value for c in at.caption)
    assert not any("· Muse tool orchestration;" in c.value for c in at.caption)
