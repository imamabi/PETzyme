"""Bounded enzyme discovery within a documented assay group; no global activity ranking."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from .activity_snapshot import iter_assays
from .activity_types import ActivitySnapshot, AssayReference
from .contracts import JSONValue, NrelTable, ToolResult
from .observations import load_nrel, parse_nrel
from .pet_retrieval import ROOT, load_activity_source
from .results import outcome

UNIT = "µmol aromatic products/mg enzyme"
MAX_DISCOVERY_ENZYMES = 5  # A bounded scientific projection, never a downloadable leaderboard.
REQUIRED_CONDITIONS = (
    "temperature_c",
    "ph",
    "duration_value",
    "duration_unit",
    "buffer",
    "enzyme_preparation",
    "enzyme_amount_value",
    "enzyme_amount_unit",
    "substrate_amount_value",
    "substrate_amount_unit",
    "reaction_volume_value",
    "reaction_volume_unit",
)
LIMITATIONS = [
    "Ranking is a COMPUTED mean of two explicitly documented biological-duplicate endpoint responses within one exact protocol/publication group; individual MEASURED values are retained.",
    "This is not a global enzyme ranking, demonstrated superiority, optimum, kinetic rate or prediction. Two duplicates do not establish statistical significance.",
    "PAZy units are explicitly recorded publication-derived curation. NREL CSV units remain UNKNOWN/UNVERIFIED and are excluded from ranking, not converted or pooled.",
    "Names denote source-labelled entries, not independently sequence-authenticated assay constructs. NREL name/condition references are candidates only; same-study curation is not replication.",
    "Only fully documented, non-digitized, uncensored biological-duplicate pairs in the supported product-per-enzyme endpoint definition are eligible. Other measurement methods remain outside this ranking.",
    "Results are transient PAZy-derived projections: no history, exports, files or caches.",
]


class DiscoveryMeasurement(TypedDict):
    value: float
    unit: str
    locator: str
    biological_duplicate: int
    evidence_category: Literal["MEASURED"]


class DiscoveryNrelReference(TypedDict):
    original_response: str
    unit_status: str
    source_row: int
    source_column: str


class DiscoveryEntry(TypedDict):
    enzyme: str
    source_protein_id: int
    response: float
    unit: str
    temperature_c: float
    ph: float
    pet_form: str
    source: str
    publication_doi: str
    assay_locator: str
    workbook_reference: str
    measurements: list[DiscoveryMeasurement]
    nrel_references: list[DiscoveryNrelReference]


class DiscoveryGroup(TypedDict):
    publication_doi: str
    temperature_c: float
    ph: float
    pet_form: str
    duration_value: float
    duration_unit: str
    unit: str
    measurement_definition: str
    protocol: dict[str, JSONValue]
    eligible_enzymes: int
    selection_rule: str


class EnzymeDiscovery(TypedDict):
    answer: str
    entries: list[DiscoveryEntry]
    group: DiscoveryGroup | None
    omitted_enzymes: int
    eligible_groups: int
    excluded_assays: int
    nrel_unit_status: str
    global_ranking_supported: Literal[False]
    persistence: Literal["TRANSIENT_PAZY_SELECTION"]


@dataclass
class _Candidate:
    reference: AssayReference
    protocol: dict[str, Any]
    measurements: list[DiscoveryMeasurement]
    aliases: set[str]


def _candidate(ref: AssayReference, aliases: set[str]) -> _Candidate | None:
    # Trust boundary: refs come only from the checksum/schema-validated snapshot loader.
    assay = cast(dict[str, Any], ref["assay"])
    c, material = assay["conditions"], assay["material"]
    if any(c.get(k) in (None, "") for k in REQUIRED_CONDITIONS):
        return None
    if not all(type(c[k]) in (int, float) for k in ("temperature_c", "ph", "duration_value")):
        return None
    if (
        c["enzyme_preparation"] != "purified"
        or c["enzyme_amount_unit"] not in ("µg", "mg", "g")
        or any(
            type(c[k]) not in (int, float) or c[k] <= 0
            for k in (
                "duration_value",
                "enzyme_amount_value",
                "substrate_amount_value",
                "reaction_volume_value",
            )
        )
    ):
        return None
    if any(
        material.get(k) in (None, "")
        for k in ("reported_name", "form", "morphology", "crystallinity_percent")
    ):
        return None
    if not any(m.get("role") == "quantitative" for m in assay["detection_methods"]) or not assay[
        "evidence"
    ].get("location"):
        return None
    measurements = []
    analytes = set()
    for index, m in enumerate(assay["measurements"]):
        if m.get("metric_type") != "specific_activity" or m.get("unit") != UNIT:
            continue
        label = re.match(r"Biological duplicate ([12]) of 2:", m.get("reported_result") or "")
        if (
            not label
            or m.get("replicate_n") != 2
            or m.get("value_is_digitized") is not False
            or m.get("qualifier") not in (None, "", "=")
            or type(m.get("value")) not in (int, float)
            or not math.isfinite(m["value"])
            or not m.get("analyte")
        ):
            return None
        analytes.add(m["analyte"])
        measurements.append(
            DiscoveryMeasurement(
                value=m["value"],
                unit=m["unit"],
                biological_duplicate=int(label[1]),
                evidence_category="MEASURED",
                locator=f"#/{ref['parent_index']}/assays/{ref['assay_index']}/measurements/{index}",
            )
        )
    if (
        len(measurements) != 2
        or {m["biological_duplicate"] for m in measurements} != {1, 2}
        or len(analytes) != 1
    ):
        return None
    # Compare mass-normalized yield at equal mass loading, not per-molecule catalysis.
    # Molar concentration differs with molecular weight; retain that distinction explicitly.
    conditions = {
        k: v
        for k, v in c.items()
        if k not in ("enzyme_concentration_value", "enzyme_concentration_unit")
    }
    protocol = {
        "conditions": conditions,
        "material": material,
        "detection_methods": assay["detection_methods"],
        "metric_type": "specific_activity",
        "analyte": next(iter(analytes)),
        "unit": UNIT,
        "replicate_definition": "Two explicitly labelled biological duplicates",
        "normalization": "Product per mg enzyme; equal documented enzyme mass. Molar concentrations are not assumed equal.",
    }
    return _Candidate(ref, protocol, measurements, aliases)


def rank_discovery_group(
    snapshot: ActivitySnapshot,
    table: NrelTable | None,
    limit: int = 5,
    temperature_c: float | None = None,
    ph: float | None = None,
    pet_form: Literal["CRYSTALLINE_POWDER", "AMORPHOUS_FILM"] | None = None,
) -> ToolResult[EnzymeDiscovery]:
    """Flow: Receives authenticated ActivitySnapshot, optional NrelTable and filters from discover_pet_enzymes(); returns transient ToolResult[EnzymeDiscovery] for discover_pet_enzymes(), without averaging across assays, conditions or publications."""
    groups: dict[str, list[_Candidate]] = defaultdict(list)
    excluded = 0
    for ref in iter_assays(snapshot):
        assay = cast(dict[str, Any], ref["assay"])
        c, material = assay["conditions"], assay["material"]
        form = (
            "CRYSTALLINE_POWDER"
            if (material.get("form"), material.get("morphology")) == ("powder", "crystalline")
            else "AMORPHOUS_FILM"
            if (material.get("form"), material.get("morphology")) == ("film", "amorphous")
            else None
        )
        if (
            form is None
            or pet_form is not None
            and form != pet_form
            or temperature_c is not None
            and c.get("temperature_c") != temperature_c
            or ph is not None
            and c.get("ph") != ph
        ):
            continue
        parent = cast(dict[str, Any], snapshot["records"][ref["parent_index"]])
        aliases = {ref["protein_name"].casefold()} | {
            s["name"].casefold() for s in parent.get("synonyms") or [] if s.get("name")
        }
        candidate = _candidate(ref, aliases)
        if candidate is None:
            excluded += 1
            continue
        key = json.dumps(
            [
                ref["publication_doi"].casefold(),
                c["temperature_c"],
                c["ph"],
                form,
                candidate.protocol,
            ],
            sort_keys=True,
        )
        groups[key].append(candidate)
    eligible = {}
    for key, candidates in groups.items():
        by_protein: dict[int, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            by_protein[candidate.reference["protein_id"]].append(candidate)
        selected = []
        for same_protein in by_protein.values():
            unique: dict[str, _Candidate] = {}
            for v in same_protein:
                signature = json.dumps(
                    [
                        v.reference["assay"]["evidence"],
                        sorted(
                            (
                                {
                                    "value": m["value"],
                                    "unit": m["unit"],
                                    "biological_duplicate": m["biological_duplicate"],
                                }
                                for m in v.measurements
                            ),
                            key=lambda m: m["biological_duplicate"],
                        ),
                    ],
                    sort_keys=True,
                )
                unique.setdefault(signature, v)
            if len(unique) != 1:
                excluded += len(same_protein)  # Never choose the best assay for a protein.
                continue
            selected.append(next(iter(unique.values())))
        unambiguous = [
            v for v in selected if not any(v is not w and v.aliases & w.aliases for w in selected)
        ]
        excluded += len(selected) - len(unambiguous)
        if unambiguous:
            eligible[key] = unambiguous
    if not eligible:
        value = EnzymeDiscovery(
            answer="No documented comparable assay group matched these filters; no ranking or unit conversion was generated.",
            entries=[],
            group=None,
            omitted_enzymes=0,
            eligible_groups=0,
            excluded_assays=excluded,
            nrel_unit_status="UNKNOWN/UNVERIFIED",
            global_ranking_supported=False,
            persistence="TRANSIENT_PAZY_SELECTION",
        )
        return outcome(
            "NO_DATA",
            value,
            [snapshot["source"]] + ([table["source"]] if table else []),
            LIMITATIONS,
            "COMPUTED",
        )
    parsed = parse_nrel(table)["result"] if table else None
    observations = parsed["observations"] if parsed else []

    def nrel_overlap(key: str) -> int:
        _, t, p, f, _ = json.loads(key)
        return sum(
            any(
                (o["enzyme_identifier"] or "").casefold() in v.aliases
                and o["temperature_c"] == t
                and o["ph"] == p
                and o["pet_form"] == f
                for o in observations
            )
            for v in eligible[key]
        )

    # Select by coverage before examining magnitudes; names are only nominal source links.
    group_key = min(eligible, key=lambda k: (-len(eligible[k]), -nrel_overlap(k), k))
    candidates = eligible[group_key]
    ranked = sorted(
        candidates,
        key=lambda v: (
            -math.fsum(m["value"] for m in v.measurements) / 2,
            v.reference["protein_name"].casefold(),
            v.reference["protein_id"],
        ),
    )
    doi, temperature, selected_ph, form, protocol = json.loads(group_key)
    entries = []
    for v in ranked[:limit]:
        ref = v.reference
        matches = [
            o
            for o in observations
            if (o["enzyme_identifier"] or "").casefold() in v.aliases
            and o["temperature_c"] == temperature
            and o["ph"] == selected_ph
            and o["pet_form"] == form
        ]
        entries.append(
            DiscoveryEntry(
                enzyme=ref["protein_name"],
                source_protein_id=ref["protein_id"],
                response=math.fsum(m["value"] for m in v.measurements) / 2,
                unit=UNIT,
                temperature_c=temperature,
                ph=selected_ph,
                pet_form=form,
                source="PAZy publication-derived curation",
                publication_doi=doi,
                assay_locator=f"#/{ref['parent_index']}/assays/{ref['assay_index']}",
                workbook_reference=str(
                    cast(dict[str, JSONValue], ref["assay"]["evidence"]).get("location")
                ),
                measurements=v.measurements,
                nrel_references=[
                    DiscoveryNrelReference(
                        original_response=o["original_text"],
                        unit_status=o["unit_status"],
                        source_row=o["source_row"],
                        source_column=o["source_column"],
                    )
                    for o in matches[:1]
                ],
            )
        )
    public_protocol = {k: protocol["conditions"][k] for k in REQUIRED_CONDITIONS}
    public_protocol.update(
        activity_archive=snapshot["source"]["url"],
        archive_sha256=snapshot["sha256"],
        PET_material=protocol["material"]["reported_name"],
        crystallinity_percent=protocol["material"]["crystallinity_percent"],
        normalization=protocol["normalization"],
        detection="; ".join(m["method"] for m in protocol["detection_methods"]),
    )
    group = DiscoveryGroup(
        publication_doi=doi,
        temperature_c=temperature,
        ph=selected_ph,
        pet_form=form,
        duration_value=protocol["conditions"]["duration_value"],
        duration_unit=protocol["conditions"]["duration_unit"],
        unit=UNIT,
        measurement_definition="Aromatic-product yield per enzyme mass at the recorded endpoint; not a kinetic rate.",
        protocol=public_protocol,
        eligible_enzymes=len(candidates),
        selection_rule="Largest eligible distinct-entry group; ties prefer nominal NREL condition coverage, then publication/temperature/pH/form/protocol JSON, never response magnitude. Name overlap is not identity proof.",
    )
    value = EnzymeDiscovery(
        answer=f"Top {len(entries)} source-labelled enzymes by mean recorded endpoint response within one documented {group['duration_value']} {group['duration_unit']} assay group: {temperature}°C, pH {selected_ph}, {form.replace('_', ' ').lower()}. This is not a global ranking or demonstrated activity superiority.",
        entries=entries,
        group=group,
        omitted_enzymes=max(0, len(candidates) - len(entries)),
        eligible_groups=len(eligible),
        excluded_assays=excluded,
        nrel_unit_status="UNKNOWN/UNVERIFIED",
        global_ranking_supported=False,
        persistence="TRANSIENT_PAZY_SELECTION",
    )
    return outcome(
        "PARTIAL",
        value,
        [snapshot["source"]] + ([table["source"]] if table else []),
        LIMITATIONS,
        "COMPUTED",
    )


def discover_pet_enzymes(
    limit: int = 5,
    temperature_c: float | None = None,
    ph: float | None = None,
    pet_form: Literal["CRYSTALLINE_POWDER", "AMORPHOUS_FILM"] | None = None,
    include_nrel_references: bool = True,
) -> ToolResult[EnzymeDiscovery]:
    """Discover up to five distinct source-labelled PET enzymes within one documented assay group.

    Filters are exact. If absent, choose the largest eligible group before ranking.
    Preserve each biological duplicate and report a COMPUTED within-assay mean;
    never pool assays or rank NREL values with unverified units.
    Flow: Receives external typed limit/condition filters from invoke_tool() or run_local_discovery(); returns transient ToolResult[EnzymeDiscovery] from rank_discovery_group() for project_evidence() and run_local_discovery().
    """
    if (
        type(include_nrel_references) is not bool
        or type(limit) is not int
        or not 1 <= limit <= MAX_DISCOVERY_ENZYMES
        or any(
            x is not None and (type(x) not in (int, float) or not math.isfinite(x))
            for x in (temperature_c, ph)
        )
        or ph is not None
        and not 0 <= ph <= 14
        or pet_form not in (None, "CRYSTALLINE_POWDER", "AMORPHOUS_FILM")
    ):
        return outcome(
            "ERROR",
            None,
            [],
            LIMITATIONS,
            "COMPUTED",
            "Supply 1–5 entries and optional finite exact PET conditions.",
        )
    loaded = load_activity_source()
    nrel = load_nrel(str(ROOT / "data/nrel/label_data.csv")) if include_nrel_references else None
    if loaded["result"] is None:
        return outcome(
            "ERROR",
            None,
            [],
            LIMITATIONS,
            "COMPUTED",
            "Pinned local scientific sources unavailable; no fallback or download.",
        )
    table = nrel["result"] if nrel is not None else None
    try:
        audit = json.loads((ROOT / "NREL_SOURCE_MANIFEST.json").read_text())["unit_reconciliation"] if include_nrel_references else {}
        if table is None or audit["csv_sha256"] != table["csv_sha256"]:
            table = None
    except (OSError, ValueError, KeyError, TypeError):
        table = None
    result = rank_discovery_group(loaded["result"], table, limit, temperature_c, ph, pet_form)
    if table is None:
        result["limitations"] = [
            *result["limitations"],
            "NREL reference/audit unavailable or unmatched; no NREL values or absence claim supplied. PAZy eligibility remains independent.",
        ]
    return result
