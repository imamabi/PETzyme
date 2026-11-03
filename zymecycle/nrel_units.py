"""Explicit working NREL unit inference with pinned, record-bound numerical validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import TypedDict, cast

from .contracts import JSONValue, Observation, UnitRepresentation

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_SHA256 = "700a78ab17348b0fdff81df5c8e49dc7e410c3bf59e0b69a2cc3f568ade30284"
SOURCE_UNIT = "mmol aromatic products/mg enzyme (INFERRED)"
DISPLAY_UNIT = "µmol aromatic products/mg enzyme"
METHOD = "U_VOLUME_ROUND_THEN_MEAN_V1"
# V is recorded at 0.001 mmol/mg precision; half a last-place unit is a
# reconstruction tolerance, not experimental uncertainty or a detection limit.
WORKBOOK_V_ROUNDING_TOLERANCE = 0.0005 + 1e-12


class ActivityRow(TypedDict):
    row: int
    sample_id: str
    pH: float
    temperature_c: float
    substrate: str
    reaction_vol_mL: float
    mM_product_per_mg_enzyme: float
    mmol_product_per_mg_enzyme: float
    umol_product_per_mg_enzyme: float


class WorkbookInspection(TypedDict):
    column_headers: dict[str, str]
    formula_examples: dict[str, str]
    formula_count: int


def read_activity_workbook(data: bytes) -> tuple[list[ActivityRow], WorkbookInspection]:
    """Flow: Receives checksum-verified in-memory workbook bytes from main(); returns ActivityRow list and WorkbookInspection for reconcile_nrel(), without saving source bytes or recalculating formulas."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(v.file_size for v in archive.infolist()) > 40_000_000:
            raise ValueError("Workbook expansion exceeds audit bound")
        strings = [
            "".join(t.itertext())
            for t in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns)
        ]
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target_by_id = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        sheet = next(
            s for s in book.findall("s:sheets/s:sheet", ns) if s.attrib["name"] == "Activity"
        )
        target = target_by_id[
            sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        ]
        path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        if ".." in path.split("/") or not path.startswith("xl/worksheets/"):
            raise ValueError("Invalid Activity worksheet path")
        tree = ET.fromstring(archive.read(path))
        rows = []
        formulas: dict[str, str] = {}
        formula_count = 0
        for row in tree.findall("s:sheetData/s:row", ns):
            values = {}
            for cell in row:
                coordinate = cell.attrib["r"]
                formula = cell.find("s:f", ns)
                if formula is not None:
                    formula_count += 1
                    if len(formulas) < 8:
                        formulas[coordinate] = formula.text or ""
                value = cell.find("s:v", ns)
                if value is not None and value.text is not None:
                    values[re.sub(r"\d+$", "", coordinate)] = (
                        strings[int(value.text)] if cell.get("t") == "s" else float(value.text)
                    )
                elif cell.get("t") == "inlineStr":
                    inline = cell.find("s:is", ns)
                    if inline is None:
                        raise ValueError("Missing inline string")
                    values[re.sub(r"\d+$", "", coordinate)] = "".join(inline.itertext())
            rows.append((int(row.attrib["r"]), values))
        headers = rows[0][1]
        required = set(ActivityRow.__required_keys__) - {"row"}
        activity: list[ActivityRow] = []
        for number, values in rows[1:]:
            named = {str(headers[k]): v for k, v in values.items() if k in headers}
            if not required <= named.keys():
                raise ValueError("Missing required Activity measurement cell")
            activity.append(cast(ActivityRow, {"row": number, **{k: named[k] for k in required}}))
        return activity, {
            "column_headers": {k: str(headers[k]) for k in ("D", "U", "V", "W")},
            "formula_examples": formulas,
            "formula_count": formula_count,
        }


def reconstruct_mmol_group(rows: list[ActivityRow]) -> float:
    """Flow: Receives matched ActivityRow list from build_unit_certificate() or reconcile_nrel(); returns float reconstructed mmol-scale mean for their cell-level validation, never an original measurement."""
    return math.fsum(
        round(r["mM_product_per_mg_enzyme"] * r["reaction_vol_mL"] / 10) / 100 for r in rows
    ) / len(rows)


class UnitCertificate(TypedDict):
    version: int
    method: str
    csv_sha256: str
    workbook_sha256: str
    paper_doi: str
    source_unit: str
    display_unit: str
    factor: int
    scope: str
    counts: dict[str, int]
    supported_cells: dict[str, str]
    failed_cells: dict[str, str]
    rounding_sensitive_cells: list[str]
    lcc_rounding_diagnostics: list[dict[str, JSONValue]]
    interpretation: str


def build_unit_certificate(csv_bytes: bytes, workbook_bytes: bytes) -> UnitCertificate:
    """Flow: Receives original CSV/workbook bytes from the explicit source-audit invocation; returns provenance-only dict for NREL_SOURCE_MANIFEST.json and standardized_nrel_values(), without storing workbook bytes or PAZy records.

    Source units are a working INFERENCE, not an explicit CSV declaration.
    The certificate records supported cell references and source worksheet positions,
    not a derivative measurement dataset. Alternate rounding disagreement is explicit.
    """
    if hashlib.sha256(workbook_bytes).hexdigest() != WORKBOOK_SHA256:
        raise ValueError("Workbook checksum mismatch; no inference authorized")
    activity, inspection = read_activity_workbook(workbook_bytes)
    grouped: dict[tuple[str, float, float, str], list[ActivityRow]] = defaultdict(list)
    for measurement in activity:
        grouped[
            (
                measurement["sample_id"],
                measurement["pH"],
                measurement["temperature_c"],
                measurement["substrate"],
            )
        ].append(measurement)
    supported: dict[str, str] = {}
    failed: dict[str, str] = {}
    sensitive: list[str] = []
    benchmarks: list[dict[str, JSONValue]] = []
    direct_count = alternate_count = total = 0
    for record, row in enumerate(csv.DictReader(io.StringIO(csv_bytes.decode())), 2):
        for column, original in row.items():
            match = re.fullmatch(
                r"activity_at_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(cryPow|aFilm)", column
            )
            if not match or not original.strip():
                continue
            total += 1
            key = f"{record}:{column}"
            ph, temperature, form = match.groups()
            hits = grouped.get((row[""], float(ph), float(temperature), form), [])
            if not hits:
                failed[key] = "No exact sample/pH/temperature/PET-form worksheet group"
                continue
            if any(
                not all(
                    math.isfinite(v)
                    for v in (
                        h["reaction_vol_mL"],
                        h["mM_product_per_mg_enzyme"],
                        h["mmol_product_per_mg_enzyme"],
                        h["umol_product_per_mg_enzyme"],
                    )
                )
                or h["reaction_vol_mL"] <= 0
                or abs(
                    h["mM_product_per_mg_enzyme"] * h["reaction_vol_mL"] / 1000
                    - h["mmol_product_per_mg_enzyme"]
                )
                > WORKBOOK_V_ROUNDING_TOLERANCE
                or not math.isclose(
                    h["mmol_product_per_mg_enzyme"] * 1000,
                    h["umol_product_per_mg_enzyme"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for h in hits
            ):
                failed[key] = "Invalid source dimensional relationship or reaction volume"
                continue
            reconstructed = reconstruct_mmol_group(hits)
            direct = math.fsum(h["mmol_product_per_mg_enzyme"] for h in hits) / len(hits)
            alternate = math.fsum(
                round(h["mmol_product_per_mg_enzyme"] * 100) / 100 for h in hits
            ) / len(hits)

            def same(value: float) -> bool:
                return math.isclose(float(original), value, rel_tol=1e-9, abs_tol=1e-12)

            direct_count += same(direct)
            alternate_count += same(alternate)
            if not same(reconstructed):
                failed[key] = "U/volume/rounding/aggregation reconstruction disagrees with CSV"
                continue
            supported[key] = ", ".join(str(h["row"]) for h in hits)
            if not same(alternate):
                sensitive.append(key)
                if row[""] == "LCC-ICCG":
                    benchmarks.append(
                        {
                            "cell": key,
                            "original": original,
                            "chosen_reconstruction": reconstructed,
                            "alternate_rounded_V": alternate,
                            "direct_V_mean": direct,
                            "workbook_rows": supported[key],
                            "status": "INFERRED_SUPPORTED; ALTERNATE_ROUNDING_DISAGREES",
                        }
                    )
    return {
        "version": 1,
        "method": METHOD,
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "workbook_sha256": WORKBOOK_SHA256,
        "paper_doi": "10.1021/acscatal.5c03460",
        "source_unit": SOURCE_UNIT,
        "display_unit": DISPLAY_UNIT,
        "factor": 1000,
        "scope": "Exact cell references in the pinned CSV; sample/pH/temperature/form worksheet linkage, not construct authentication.",
        "counts": {
            "observations": total,
            "supported": len(supported),
            "failed": len(failed),
            "direct_V_matches": direct_count,
            "rounded_V_matches": alternate_count,
            "rounding_sensitive": len(sensitive),
            "stored_formulas": inspection["formula_count"],
        },
        "supported_cells": supported,
        "failed_cells": failed,
        "rounding_sensitive_cells": sensitive,
        "lcc_rounding_diagnostics": benchmarks,
        "interpretation": "Owner-adopted evidence-backed working unit inference from source dimensions and reproduced cell arithmetic. CSV units/generating code are not author-authenticated. Converted CSV values need not equal unrounded workbook means. No cross-source pooling or identity proof.",
    }


def standardized_nrel_values(observations: list[Observation]) -> list[UnitRepresentation]:
    """Flow: Receives original Observation list from investigate_pet_evidence() or public_tool_update(); returns UnitRepresentation list for their explicit COMPUTED display, requiring pinned certificate, actual cell/text and source hash agreement.

    No network, cache, file writes, source edits or changes to original unit_status.
    Missing/contradictory validation leaves the affected representation unresolved.
    """
    if not observations:
        return []
    certificate = None
    original_cells: dict[str, tuple[str, str]] = {}
    try:
        certificate = json.loads((ROOT / "NREL_SOURCE_MANIFEST.json").read_text())[
            "working_unit_interpretation"
        ]
        if (
            not isinstance(certificate, dict)
            or not isinstance(certificate.get("supported_cells"), dict)
            or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in certificate["supported_cells"].items()
            )
            or not isinstance(certificate.get("failed_cells"), dict)
            or not isinstance(certificate.get("rounding_sensitive_cells"), list)
        ):
            raise ValueError("Malformed unit-validation metadata")
        content = (ROOT / "data/nrel/label_data.csv").read_bytes()
        if (
            certificate["version"] != 1
            or certificate["method"] != METHOD
            or certificate["workbook_sha256"] != WORKBOOK_SHA256
            or hashlib.sha256(content).hexdigest() != certificate["csv_sha256"]
        ):
            raise ValueError("Source/certificate mismatch")
        original_cells = {
            f"{i}:{column}": (value, row[""])
            for i, row in enumerate(csv.DictReader(io.StringIO(content.decode())), 2)
            for column, value in row.items()
            if column.startswith("activity_at_") and value.strip()
        }
    except (OSError, ValueError, KeyError, TypeError):
        certificate = None
    values = []
    for o in observations:
        key = f"{o['source_row']}:{o['source_column']}"
        condition = re.fullmatch(
            r"activity_at_([\d.]+)_([\d.]+)_(cryPow|aFilm)", o["source_column"]
        )
        matches_condition = bool(
            condition
            and float(condition[1]) == o["ph"]
            and float(condition[2]) == o["temperature_c"]
            and {"cryPow": "CRYSTALLINE_POWDER", "aFilm": "AMORPHOUS_FILM"}[condition[3]]
            == o["pet_form"]
        )
        passed = bool(
            certificate
            and o["csv_sha256"] == certificate["csv_sha256"]
            and original_cells.get(key) == (o["original_text"], o["enzyme_identifier"])
            and matches_condition
            and key in certificate["supported_cells"]
            and key not in certificate["failed_cells"]
        )
        values.append(
            UnitRepresentation(
                status="INFERRED_SUPPORTED" if passed else "UNRESOLVED",
                source_unit=SOURCE_UNIT if passed else "UNKNOWN/UNVERIFIED",
                display_unit=DISPLAY_UNIT if passed else "UNKNOWN/UNVERIFIED",
                standardized_value=float(Decimal(o["original_text"]) * 1000) if passed else None,
                conversion_factor=1000 if passed else None,
                original_declaration="UNKNOWN/UNVERIFIED",
                evidence_category="COMPUTED",
                workbook_rows=certificate["supported_cells"][key] if passed and certificate else "",
                method=METHOD,
                rounding_sensitive=bool(
                    certificate and key in certificate["rounding_sensitive_cells"]
                ),
                reason=(
                    "Cell reconstruction validated; source units INFERRED, not explicitly declared in CSV. "
                    "Rounding/aggregation can differ from unrounded published measurements."
                    if passed
                    else "Missing or failed source/cell validation; original value retained without conversion."
                ),
            )
        )
    return values


def workbook_reference_link(reference_rows: str, doi: str, locator: str | None) -> str:
    """Flow: Receives validated worksheet-row str and actual PAZy DOI/locator from investigate_pet_evidence(); returns str documentary linkage for its transient results, never sequence or experimental construct identity."""
    if (
        doi.casefold() != "10.1021/acscatal.5c03460"
        or not locator
        or "cs5c03460_si_002.xlsx" not in locator
    ):
        return "UNVERIFIED"
    match = re.search(r"Activity sheet, rows? ([0-9, ]+)", locator, re.I)
    if not match:
        return "UNVERIFIED"
    expected = {int(v) for v in re.findall(r"\d+", reference_rows)}
    cited = {int(v) for v in re.findall(r"\d+", match[1])}
    if expected and expected == cited:
        return "DOCUMENTED_WORKBOOK_ROW_MATCH; not construct identity or independent replication"
    return "PARTIAL_ROW_OVERLAP; aggregation unresolved" if expected & cited else "UNVERIFIED"
