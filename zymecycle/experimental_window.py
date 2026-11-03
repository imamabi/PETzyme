"""Explicit researcher limits, independent of observations and feasibility claims."""

from __future__ import annotations

import math
import re

from .contracts import Condition, ExperimentalWindow, PetForm

TEMPERATURE = r"(?<![A-Za-z0-9_])(-?\d+(?:\.\d+)?)\s*(?:°\s*C|degrees?\s*C|celsius|C)\b"
MAX_PH = r"(?:max(?:imum)?\s+pH\s*(?:is|of|=|:)?\s*|pH\s*(?:<=|≤|(?:cannot|must\s+not|does\s+not)\s+exceed)\s*|(?:cannot\s+(?:operate\s+)?(?:above|exceed)\s+pH\s*))(\d+(?:\.\d+)?)|pH\s*(\d+(?:\.\d+)?)\s*(?:or\s+(?:below|less|lower)|maximum)"
MIN_PH = r"(?:min(?:imum)?\s+pH\s*(?:is|of|=|:)?\s*|pH\s*(?:>=|≥)\s*)(\d+(?:\.\d+)?)|pH\s*(\d+(?:\.\d+)?)\s*or\s+(?:above|more)"


def parse_experimental_window(question: str) -> ExperimentalWindow:
    """Flow: Receives external question str from interpret_request(); returns ExperimentalWindow for its exact filters and advisor restrictions, keeping pH bounds distinct from measured setpoints."""
    bounds = []
    for pattern in (MIN_PH, MAX_PH):
        matches = list(re.finditer(pattern, question, re.I))
        values = {float(next(g for g in m.groups() if g is not None)) for m in matches}
        if len(values) > 1:
            raise ValueError(
                "Conflicting pH bounds require clarification; no limit was overwritten."
            )
        bounds.append((next(iter(values), None), matches))
    ph = [
        float(m.group(1))
        for m in re.finditer(r"\bpH\s*(\d+(?:\.\d+)?)", question, re.I)
        if not any(b.start() <= m.start() < b.end() for _, matches in bounds for b in matches)
    ]
    temperatures = list(dict.fromkeys(float(x) for x in re.findall(TEMPERATURE, question, re.I)))
    only = re.search(r"\bonly\b(?:(?![!?;\n]|\.(?!\d)).){0,65}", question, re.I)
    if only and re.search(TEMPERATURE, only.group(), re.I):
        temperatures = list(
            dict.fromkeys(float(x) for x in re.findall(TEMPERATURE, only.group(), re.I))
        )
    forms: list[PetForm] = []
    if re.search(r"\bpowder\b", question, re.I):
        forms.append("CRYSTALLINE_POWDER")
    if re.search(r"\bfilm\b", question, re.I):
        forms.append("AMORPHOUS_FILM")
    window: ExperimentalWindow = {
        "temperatures_c": temperatures,
        "ph_values": list(dict.fromkeys(ph)),
        "minimum_ph": bounds[0][0],
        "maximum_ph": bounds[1][0],
        "pet_forms": forms,
    }
    if not valid_window(window):
        raise ValueError(
            "Experimental bounds or exact pH values conflict; clarify them before planning."
        )
    return window


def valid_window(value: object) -> bool:
    """Flow: Receives an external window object from advise_next_experiment(); returns bool for its trusted constraint boundary, never laboratory feasibility."""
    if not isinstance(value, dict) or set(value) != {
        "temperatures_c",
        "ph_values",
        "minimum_ph",
        "maximum_ph",
        "pet_forms",
    }:
        return False

    def number(n: object) -> bool:
        return isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n)

    if any(
        not isinstance(value[k], list) or len(value[k]) > 8 or any(not number(n) for n in value[k])
        for k in ("temperatures_c", "ph_values")
    ):
        return False
    if any(
        value[k] is not None and (not number(value[k]) or not 0 <= value[k] <= 14)
        for k in ("minimum_ph", "maximum_ph")
    ):
        return False
    low, high = value["minimum_ph"], value["maximum_ph"]
    return (
        not (low is not None and high is not None and low > high)
        and all(
            0 <= n <= 14 and (low is None or n >= low) and (high is None or n <= high)
            for n in value["ph_values"]
        )
        and isinstance(value["pet_forms"], list)
        and len(value["pet_forms"]) <= 2
        and all(f in ("CRYSTALLINE_POWDER", "AMORPHOUS_FILM") for f in value["pet_forms"])
    )


def within_window(condition: Condition, window: ExperimentalWindow | None) -> bool:
    """Flow: Receives source-derived Condition and validated window from _coverage_advice(); returns bool for proposal eligibility without treating bounds as observed or feasible conditions."""
    return window is None or (
        (not window["temperatures_c"] or condition["temperature_c"] in window["temperatures_c"])
        and (not window["ph_values"] or condition["ph"] in window["ph_values"])
        and (window["minimum_ph"] is None or condition["ph"] >= window["minimum_ph"])
        and (window["maximum_ph"] is None or condition["ph"] <= window["maximum_ph"])
        and (not window["pet_forms"] or condition["pet_form"] in window["pet_forms"])
    )
