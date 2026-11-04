"""One transient experimental record contract for scientific consumers and UI.

Only existing allowlisted source projections enter this adapter. Missing metadata
stays None; names, publications and numeric agreement never authenticate constructs.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Literal, NotRequired, TypedDict

from .contracts import JSONValue, UnitRepresentation
from .experimental_coverage import ExperimentalCoverage


class NrelDisplayObservation(TypedDict):
    working_units: NotRequired[UnitRepresentation]
    source_row_links: NotRequired[list[dict[str, str]]]
    enzyme_identifier: str
    pet_form: str
    temperature_c: float
    ph: float
    original_text: str
    unit_status: str
    source_row: int
    source_column: str


class NrelDisplaySelection(TypedDict):
    observations: list[NrelDisplayObservation]
    total_matches: NotRequired[int | None]


class DisplayMeasurement(TypedDict):
    record: dict[str, JSONValue]
    locator: str


class DisplayAssay(TypedDict):
    locator: NotRequired[str]
    evidence_locator: NotRequired[str | None]
    identity_status: NotRequired[str]
    detection_methods: NotRequired[list[JSONValue]]
    protein_name: str
    conditions: dict[str, JSONValue]
    material: dict[str, JSONValue]
    publication_doi: str
    measurements: list[DisplayMeasurement]


class PazyDisplaySelection(TypedDict):
    records: list[DisplayAssay]
    total_matches: NotRequired[int | None]
    total_matching_measurements: NotRequired[int | None]


class CombinedExperimentalPreview(TypedDict):
    comparisons: NotRequired[list[dict[str, JSONValue]]]
    nrel: NrelDisplaySelection
    pazy: PazyDisplaySelection
    experimental_coverage: NotRequired[ExperimentalCoverage | None]
    experimental_coverages: NotRequired[list[ExperimentalCoverage]]
    units: NotRequired[dict[str, JSONValue]]


class ExperimentalRecord(TypedDict):
    comparability_assessments: list[dict[str, JSONValue]]
    enzyme: str
    source: Literal["NREL", "PAZy"]
    publication: str | None
    reference: str
    experiment_identity: Literal["SOURCE_REFERENCE_ONLY"]
    pet_form: str
    crystallinity_percent: JSONValue
    temperature_c: JSONValue
    ph: JSONValue
    duration_value: JSONValue
    duration_unit: JSONValue
    protocol: dict[str, JSONValue]
    original_response: str
    original_unit: str
    qualified_unit: UnitRepresentation | None
    measurement_definition: str
    replicate_information: dict[str, JSONValue]
    identity_status: str
    comparability_status: Literal["NOT_AUTHENTICATED"]
    evidence_category: Literal["MEASURED"]
    row_linkage: str


def reported_response(record: dict[str, JSONValue]) -> str:
    """Flow: Receives an allowlisted measurement record from experimental_records(); returns its original qualified response str for that adapter and measurement_summary(), without averaging or unit inference."""
    value = record.get("value")
    if value is None:
        value = next((v for v in (record.get("reported_result"), record.get("categorical_result"))
                      if isinstance(v, str) and v.strip()), None)
        if value is None:
            return "Not recorded"
        # A textual source statement is not a recorded numeric measurement.
        # Preserve numeric-looking text without presenting it as a numeric zero.
        try:
            Decimal(value.strip())
        except InvalidOperation:
            pass
        else:
            return "Reported text: " + value
    elif (isinstance(value, bool) or not isinstance(value, (int, float))
          or isinstance(value, float) and not math.isfinite(value)):
        return "Unavailable (invalid recorded value)"
    qualifier = record.get("qualifier")
    return (str(qualifier) + " " if qualifier else "") + str(value)


def experimental_records(value: CombinedExperimentalPreview) -> list[ExperimentalRecord]:
    """Flow: Receives validated bounded CombinedExperimentalPreview from preview_pet_evidence() or accumulate_pet_results(); returns list[ExperimentalRecord] for _coverage_advice() and combined_experimental_rows(), preserving source measurement definitions and qualified units."""
    records: list[ExperimentalRecord] = []
    for o in value["nrel"]["observations"]:
        working = o.get("working_units")
        definition = (
            "Original CSV response; original CSV declaration unverified (UNKNOWN/UNVERIFIED)."
        )
        if working and working["status"] == "INFERRED_SUPPORTED":
            definition += (
                " COMPUTED inferred unit representation (Inferred from source reconciliation)."
            )
            if working["rounding_sensitive"]:
                definition += " Rounding-sensitive cell; alternate worksheet rounding disagrees, see provenance."
        records.append(
            {
                "enzyme": o["enzyme_identifier"],
                "source": "NREL",
                "publication": "10.1021/acscatal.5c03460",
                "reference": f"NREL CSV record {o['source_row']}, {o['source_column']}",
                "experiment_identity": "SOURCE_REFERENCE_ONLY",
                "pet_form": o["pet_form"].replace("_", " ").lower(),
                "crystallinity_percent": None,
                "temperature_c": o["temperature_c"],
                "ph": o["ph"],
                "duration_value": None,
                "duration_unit": None,
                "protocol": {},
                "original_response": o["original_text"],
                "original_unit": o["unit_status"],
                "qualified_unit": working,
                "measurement_definition": definition,
                "replicate_information": {},
                "identity_status": "UNVERIFIED_CONSTRUCT",
                "comparability_status": "NOT_AUTHENTICATED",
                "evidence_category": "MEASURED",
                "row_linkage": "; ".join(link["status"] for link in o.get("source_row_links", [])),
                "comparability_assessments": [
                    c
                    for c in value.get("comparisons", [])
                    if c.get("nrel_record") == o["source_row"]
                    and c.get("nrel_column") == o["source_column"]
                ],
            }
        )
    for card in value["pazy"]["records"]:
        c, m = card["conditions"], card["material"]
        for measurement in card["measurements"]:
            r = measurement["record"]
            records.append(
                {
                    "enzyme": card["protein_name"],
                    "source": "PAZy",
                    "publication": card["publication_doi"],
                    "reference": card["publication_doi"] + " · " + measurement["locator"],
                    "experiment_identity": "SOURCE_REFERENCE_ONLY",
                    "pet_form": " ".join(str(m[k]) for k in ("morphology", "form") if m.get(k))
                    or "Not documented",
                    "crystallinity_percent": m.get("crystallinity_percent"),
                    "temperature_c": c.get("temperature_c"),
                    "ph": c.get("ph"),
                    "duration_value": c.get("duration_value"),
                    "duration_unit": c.get("duration_unit"),
                    "protocol": {**c, "detection_methods": card.get("detection_methods", [])},
                    "original_response": reported_response(r),
                    "original_unit": str(r.get("unit") or "UNKNOWN/UNVERIFIED"),
                    "qualified_unit": None,
                    "measurement_definition": str(r.get("metric_type"))
                    + " · "
                    + str(r.get("analyte")),
                    "replicate_information": {
                        k: r.get(k)
                        for k in (
                            "replicate_n",
                            "replicate_n_reported",
                            "uncertainty_value",
                            "uncertainty_type",
                            "reported_result",
                        )
                    },
                    "identity_status": card.get("identity_status", "UNVERIFIED_CONSTRUCT"),
                    "comparability_status": "NOT_AUTHENTICATED",
                    "evidence_category": "MEASURED",
                    "row_linkage": card.get("evidence_locator") or "",
                    "comparability_assessments": [
                        x
                        for x in value.get("comparisons", [])
                        if x.get("pazy_locator") == measurement["locator"]
                    ],
                }
            )
    return records
