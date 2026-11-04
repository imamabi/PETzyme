"""Nominal condition coverage from authenticated local sources, never pooled responses."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import islice, product
from typing import Literal, TypedDict

from .activity_types import AssayReference
from .contracts import Observation
from .cross_overlap import NREL_PAPER

# Bound transient projections, not the source scan or scientific claim of coverage.
MAX_COVERAGE_CONDITIONS = 24
MAX_MISSING_CONDITIONS = 8


def _number(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _label(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


class CoverageCondition(TypedDict):
    temperature_c: float | None
    ph: float | None
    form: str
    morphology: str
    nrel_observations: int
    pazy_assays: int
    pazy_measurements: int
    pazy_publications: int


class ExperimentalCoverage(TypedDict):
    source_scope: Literal['BOTH', 'PAZY_ONLY']
    enzyme: str
    nrel_observations: int
    pazy_assays: int
    pazy_measurements: int
    pazy_publications: int
    nrel_candidate_records: int
    pazy_candidate_records: int
    same_publication_assays: int
    other_publication_assays: int
    nominal_overlap: int
    pazy_additional_conditions: int
    nrel_only_conditions: int
    conditions: list[CoverageCondition]
    conditions_omitted: int
    domain_temperature_count: int
    domain_ph_count: int
    domain_material_count: int
    domain_combinations: int
    documented_combinations: int
    records_with_unknown_condition: int
    missing_in_both: list[CoverageCondition]
    missing_in_both_count: int
    missing_omitted: int
    basis: str


def summarize_experimental_coverage(
    enzyme: str, observations: list[Observation], assays: list[AssayReference],
    *, source_scope: Literal['BOTH', 'PAZY_ONLY'] = 'BOTH'
) -> ExperimentalCoverage:
    """Flow: Receives an exact candidate label str and scoped Observation/AssayReference lists from investigate_pet_evidence(); returns transient ExperimentalCoverage for preview_pet_evidence() and coverage-aware advisor presentation.

    Counts only nominal conditions; names never establish sequence/construct identity.
    Missing combinations mean absent in these selected sources within their observed
    temperature/pH/form domain, never globally unexplored or scientifically feasible.
    Morphology is part of the key; unknown morphology cannot match crystalline PET.
    """
    grouped: dict[tuple[float | None, float | None, str, str], CoverageCondition] = {}
    publications: dict[tuple[float | None, float | None, str, str], set[str]] = defaultdict(set)

    def cell(key: tuple[float | None, float | None, str, str]) -> CoverageCondition:
        if key not in grouped:
            grouped[key] = CoverageCondition(
                temperature_c=key[0],
                ph=key[1],
                form=key[2],
                morphology=key[3],
                nrel_observations=0,
                pazy_assays=0,
                pazy_measurements=0,
                pazy_publications=0,
            )
        return grouped[key]

    for observation in observations:
        powder = observation["pet_form"] == "CRYSTALLINE_POWDER"
        item = cell(
            (
                observation["temperature_c"],
                observation["ph"],
                "powder" if powder else "film",
                "crystalline" if powder else "amorphous",
            )
        )
        item["nrel_observations"] += 1
    measurement_count = 0
    for reference in assays:
        assay = reference["assay"]
        condition, material = assay["conditions"], assay["material"]
        measurements = assay["measurements"]
        if (
            not isinstance(condition, dict)
            or not isinstance(material, dict)
            or not isinstance(measurements, list)
        ):
            raise ValueError(
                "Coverage requires validated assay conditions, material and measurements"
            )
        key = (
            _number(condition.get("temperature_c")),
            _number(condition.get("ph")),
            _label(material.get("form")),
            _label(material.get("morphology")),
        )
        item = cell(key)
        item["pazy_assays"] += 1
        item["pazy_measurements"] += len(measurements)
        measurement_count += len(measurements)
        publications[key].add(reference["publication_doi"].casefold())
        item["pazy_publications"] = len(publications[key])
    cells = list(grouped.values())
    temperatures = sorted({k[0] for k in grouped if k[0] is not None})
    ph_values = sorted({k[1] for k in grouped if k[1] is not None})
    materials = sorted({(k[2], k[3]) for k in grouped if k[2] != "unknown" and k[3] != "unknown"})
    known_keys = {
        k for k in grouped if k[0] is not None and k[1] is not None and (k[2], k[3]) in materials
    }
    missing_count = len(temperatures) * len(ph_values) * len(materials) - len(known_keys)
    missing = []
    candidates = ((t, p, f, m) for t, p, (f, m) in product(temperatures, ph_values, materials))
    for key in islice((k for k in candidates if k not in grouped), MAX_MISSING_CONDITIONS):
        missing.append(
            CoverageCondition(
                temperature_c=key[0],
                ph=key[1],
                form=key[2],
                morphology=key[3],
                nrel_observations=0,
                pazy_assays=0,
                pazy_measurements=0,
                pazy_publications=0,
            )
        )
    return ExperimentalCoverage(
        source_scope=source_scope,
        enzyme=enzyme,
        nrel_observations=len(observations),
        pazy_assays=len(assays),
        pazy_measurements=measurement_count,
        pazy_publications=len({r["publication_doi"].casefold() for r in assays}),
        nrel_candidate_records=len({(o["csv_sha256"], o["source_row"]) for o in observations}),
        pazy_candidate_records=len({r["protein_id"] for r in assays}),
        same_publication_assays=sum(r["publication_doi"].casefold() == NREL_PAPER for r in assays),
        other_publication_assays=sum(r["publication_doi"].casefold() != NREL_PAPER for r in assays),
        nominal_overlap=sum(bool(c["nrel_observations"] and c["pazy_assays"]) for c in cells),
        pazy_additional_conditions=sum(
            bool(c["pazy_assays"] and not c["nrel_observations"]) for c in cells
        ),
        nrel_only_conditions=sum(
            bool(c["nrel_observations"] and not c["pazy_assays"]) for c in cells
        ),
        conditions=cells[:MAX_COVERAGE_CONDITIONS],
        conditions_omitted=max(0, len(cells) - MAX_COVERAGE_CONDITIONS),
        domain_temperature_count=len(temperatures),
        domain_ph_count=len(ph_values),
        domain_material_count=len(materials),
        domain_combinations=len(temperatures) * len(ph_values) * len(materials),
        documented_combinations=len(known_keys),
        records_with_unknown_condition=sum(
            c["nrel_observations"] + c["pazy_assays"] for key, c in grouped.items()
            if key not in known_keys
        ),
        missing_in_both=missing,
        missing_in_both_count=missing_count,
        missing_omitted=max(0, missing_count - len(missing)),
        basis="Exact candidate label/alias and nominal temperature/pH/form/morphology only; "
        "identity and original-experiment linkage remain unverified. Inventory spans all conditions "
        "for this label within the selected publication/protocol scope, not just the response preview. "
        "Shared publication is not independent replication. Missing combinations refer only to "
        "the Cartesian product of separately observed non-missing temperatures, pH values and material/form pairs in these pinned sources, not the whole literature. Unknown condition fields are excluded from the grid and counted separately. "
        "No response pooling, ranking, conversion or feasibility inference.",
    )
