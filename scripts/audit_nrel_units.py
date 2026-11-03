"""Explicit, read-only NREL source audit; supplementary workbook bytes remain in memory.

Run manually with the existing Python/httpx environment, not during application use.
Default mode tests a mapping hypothesis. Explicit --adopt-working-inference writes
only cell-reference validation metadata for labelled COMPUTED representations;
it never changes the original CSV or saves workbook bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import httpx

from petzyme.nrel_units import (
    ActivityRow,
    build_unit_certificate,
    read_activity_workbook,
    reconstruct_mmol_group,
)

WORKBOOK_URL = "https://ndownloader.figshare.com/files/57638729"
WORKBOOK_MD5 = "67d8c85e6c29ce21de0d5dbf77c427e1"
WORKBOOK_SHA256 = "700a78ab17348b0fdff81df5c8e49dc7e410c3bf59e0b69a2cc3f568ade30284"








def reconcile_nrel(csv_text: str, activity: list[ActivityRow]) -> dict:
    """Flow: Receives original CSV str and in-memory list[ActivityRow] from main(); returns bounded COMPUTED audit dict for main() output, never converted scientific observations."""
    groups = defaultdict(list)
    for row in activity:
        groups[(row["sample_id"], row["pH"], row["temperature_c"], row["substrate"])].append(row)
    counts = {
        "observations": 0,
        "matched": 0,
        "unmatched": 0,
        "exact_mean_matches": 0,
        "within_rounding_interval": 0,
        "outside_rounding_interval": 0,
        "workbook_scale_consistent_rows": 0,
        "reconstructed_matches": 0,
        "reconstructed_mismatches": 0,
        "rounded_V_mean_matches": 0,
    }
    for row in activity:
        counts["workbook_scale_consistent_rows"] += int(
            math.isclose(
                row["mmol_product_per_mg_enzyme"] * 1000,
                row["umol_product_per_mg_enzyme"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
    examples, exceptions, benchmark_checks = [], [], []
    for record, row in enumerate(csv.DictReader(io.StringIO(csv_text)), 2):
        for column, original in row.items():
            match = re.fullmatch(
                r"activity_at_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(cryPow|aFilm)", column
            )
            if match is None or not original.strip():
                continue
            ph, temperature, form = match.groups()
            counts["observations"] += 1
            hits = groups[(row[""], float(ph), float(temperature), form)]
            if not hits:
                counts["unmatched"] += 1
                continue
            counts["matched"] += 1
            mm_mean = math.fsum(h["mmol_product_per_mg_enzyme"] for h in hits) / len(hits)
            um_mean = math.fsum(h["umol_product_per_mg_enzyme"] for h in hits) / len(hits)
            # Independent reconstruction hypothesis from U (mM/mg) and D (mL).
            # mM = mmol/L, so U * D / 1000 = mmol/mg. Round source rows, then average.
            reconstructed = reconstruct_mmol_group(hits)
            reconstructed_match = math.isclose(
                float(original), reconstructed, rel_tol=1e-9, abs_tol=1e-12
            )
            counts[
                "reconstructed_matches" if reconstructed_match else "reconstructed_mismatches"
            ] += 1
            rounded_v_mean = math.fsum(
                round(h["mmol_product_per_mg_enzyme"] * 100) / 100 for h in hits
            ) / len(hits)
            rounded_v_match = math.isclose(
                float(original), rounded_v_mean, rel_tol=1e-9, abs_tol=1e-12
            )
            counts["rounded_V_mean_matches"] += int(rounded_v_match)
            delta = float(original) * 1000 - um_mean
            exact = math.isclose(float(original), mm_mean, rel_tol=1e-9, abs_tol=1e-12)
            # Diagnostic interval from apparent 0.01 mmol/mg rounding, not an uncertainty estimate.
            consistent = abs(delta) <= 5 + 1e-9
            counts["exact_mean_matches"] += int(exact)
            counts["within_rounding_interval" if consistent else "outside_rounding_interval"] += 1
            detail = {
                "identifier": row[""],
                "csv_record": record,
                "csv_column": column,
                "original_text": original,
                "workbook_rows": [h["row"] for h in hits],
                "n_source_rows": len(hits),
                "workbook_mmol_mean": mm_mean,
                "workbook_umol_mean": um_mean,
                "reconstructed_mmol_mean": reconstructed,
                "hypothetical_csv_times_1000": float(original) * 1000,
                "difference_umol": delta,
            }
            if not consistent and len(exceptions) < 8:
                exceptions.append(detail)
            if row[""] == "ESM050" and float(ph) == 6.5 and form == "cryPow":
                examples.append(
                    {
                        **detail,
                        "workbook_mmol_values": [h["mmol_product_per_mg_enzyme"] for h in hits],
                        "workbook_umol_values": [h["umol_product_per_mg_enzyme"] for h in hits],
                        "workbook_concentration_normalized_values": [
                            h["mM_product_per_mg_enzyme"] for h in hits
                        ],
                        "workbook_reaction_volumes_ml": [h["reaction_vol_mL"] for h in hits],
                    }
                )
            if row[""] == "LCC-ICCG":
                if not rounded_v_match:
                    benchmark_checks.append({**detail, "rounded_V_mean": rounded_v_mean})
    return {
        "evidence_category": "COMPUTED",
        "unit_status": "UNKNOWN/UNVERIFIED",
        "counts": counts,
        "matching_key": ["sample_id", "pH", "temperature_c", "substrate"],
        "aggregation_hypothesis": "Unweighted mean of all matching Activity rows; not a verified CSV-generation procedure.",
        "tolerance": "Absolute difference <= 5 umol/mg plus 1e-9 arithmetic tolerance; diagnostic rounding interval, not measurement uncertainty.",
        "exceptions": exceptions,
        "exceptions_omitted": max(0, counts["outside_rounding_interval"] - len(exceptions)),
        "selected_examples": examples,
        "benchmark_rounding_diagnostics": benchmark_checks,
        "reconstruction": "mean(round(U * D / 1000, 2)) grouped by sample_id, pH, temperature_c and substrate; U=mM_product_per_mg_enzyme, D=reaction_vol_mL. Half-even rounding implemented as round(x*100)/100.",
        "limitations": [
            "Numerical consistency does not authenticate the CSV unit definition or source-row aggregation.",
            "Workbook precision, rounding order, benchmark grouping and source pipeline require reconciliation.",
            "Zeros cannot identify a scaling factor. Missing cells are excluded, never replaced by zero.",
            "No source value changed; no converted observation is published.",
        ],
    }


def main() -> None:
    """Flow: Receives the existing local NREL CSV and explicit CLI invocation; retrieves cited supplement bytes in memory and prints a bounded reconcile_nrel() audit dict as final output."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--adopt-working-inference', action='store_true',
                        help='Write only pinned cell-validation provenance to the existing NREL manifest; never save workbook bytes.')
    args = parser.parse_args()
    data = b""
    with httpx.stream("GET", WORKBOOK_URL, timeout=25, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            data += chunk
            if len(data) > 2_000_000:
                raise ValueError("Supplement exceeds audit byte bound")
    if (
        len(data) > 2_000_000
        or hashlib.md5(data).hexdigest() != WORKBOOK_MD5
        or hashlib.sha256(data).hexdigest() != WORKBOOK_SHA256
    ):
        raise ValueError("Supplement size or published checksum mismatch")
    activity, inspection = read_activity_workbook(data)
    root = Path(__file__).resolve().parents[1]
    csv_data = (root / "data/nrel/label_data.csv").read_bytes()
    audit = reconcile_nrel(csv_data.decode(), activity)
    audit.update(
        workbook_url=WORKBOOK_URL,
        workbook_md5=WORKBOOK_MD5,
        workbook_sha256=hashlib.sha256(data).hexdigest(),
        csv_sha256=hashlib.sha256(csv_data).hexdigest(),
        workbook_activity_rows=len(activity),
        workbook_inspection=inspection,
    )
    if args.adopt_working_inference:
        certificate = build_unit_certificate(csv_data, data)
        path = root / 'NREL_SOURCE_MANIFEST.json'
        manifest = json.loads(path.read_text())
        manifest['working_unit_interpretation'] = certificate
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'adopted_counts': certificate['counts'],
                          'lcc_rounding_diagnostics': certificate['lcc_rounding_diagnostics'],
                          'failed_cells': certificate['failed_cells']}, indent=2))
    else:
        print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
