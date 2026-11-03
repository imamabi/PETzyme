"""Display-only separation of documented publication units and unresolved CSV units."""

from __future__ import annotations

from typing import TypedDict

from .nrel_units import METHOD


class UnitContext(TypedDict):
    publication: str
    workbook: str
    csv_inference: str
    record_verification: str
    source_url: str


def unit_context(manifest: object) -> UnitContext:
    """Flow: Receives original NREL manifest JSON object from render_findings(); returns UnitContext for researcher-facing unit labels, never converted observations or newly authenticated record units."""
    audit = manifest.get("unit_reconciliation", {}) if isinstance(manifest, dict) else {}
    known = isinstance(audit, dict) and audit.get("paper_doi") == "10.1021/acscatal.5c03460"
    working = manifest.get("working_unit_interpretation", {}) if isinstance(manifest, dict) else {}
    adopted = isinstance(working, dict) and working.get("method") == METHOD
    return {
        "publication": "Publication: µmol aromatic products per mg enzyme after 48 h (Figure 2); this is not an explicit unit declaration for each condensed CSV cell."
        if known
        else "Publication-level unit documentation unavailable in the supplied manifest.",
        "workbook": "Workbook: separate mM product/mg, mmol product/mg and µmol product/mg columns, with reaction volume in mL; source calculation and rounding must remain distinct from CSV attribution."
        if known
        else "Workbook unit documentation unavailable.",
        "csv_inference": (
            "Condensed CSV: mmol aromatic products/mg enzyme is INFERRED from matched worksheet dimensions, rounding and aggregation. Validated cells have an explicit COMPUTED ×1000 representation in µmol/mg; this working hypothesis is not an author-authenticated CSV declaration."
            if adopted
            else "Condensed CSV: no validated working-unit inference supplied. No automatic ×1000 conversion."
        ),
        "record_verification": "Record-level CSV units: UNKNOWN/UNVERIFIED. Original values are preserved; numerical agreement does not authenticate constructs, assay metadata or measurement uncertainty.",
        "source_url": "https://doi.org/10.1021/acscatal.5c03460",
    }
