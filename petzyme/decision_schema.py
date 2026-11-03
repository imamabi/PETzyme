"""Bounded structural decision contract; contains no scientific acceptance rules."""

from __future__ import annotations

import json
import math
import re
from typing import Literal, cast, get_args, get_origin, get_type_hints, is_typeddict

from .advisor_candidates import CONTROLS
from .contracts import JSONValue
from .decision_types import (
    Control,
    DecisionContractDiagnostics,
    DecisionSchemaCheck,
    DecisionSchemaIssue,
    DecisionSelection,
    SchemaIssueCode,
)

MAX_OPTIONS = 3
MAX_SCHEMA_ISSUES = 64  # Fixed field vocabulary and bounded arrays; never retain rejected values.
MAX_REFERENCE_ITEMS = 100
MAX_PREFERENCES = 6
OPTION_ID_PATTERN = r"P[1-9]"


def _schema(annotation: object) -> dict[str, JSONValue]:
    if is_typeddict(annotation):
        hints = get_type_hints(annotation)
        return {
            "type": "object",
            "properties": {k: _schema(t) for k, t in hints.items()},
            "required": list(hints),
            "additionalProperties": False,
        }
    origin = get_origin(annotation)
    if origin is Literal:
        return {"type": "string", "enum": list(get_args(annotation))}
    if origin is list:
        return {
            "type": "array",
            "items": _schema(get_args(annotation)[0]),
            "maxItems": MAX_REFERENCE_ITEMS,
        }
    if annotation is str:
        return {"type": "string"}
    if annotation is float:
        return {"type": "number"}
    raise TypeError("Unsupported internal decision annotation")


def decision_json_schema() -> dict[str, JSONValue]:
    """Flow: Receives internal DecisionSelection annotations; returns JSON schema dict[str, JSONValue] for decision_contract_message() and inspect_decision_schema()."""
    schema = _schema(DecisionSelection)
    properties = cast(dict[str, dict[str, JSONValue]], schema["properties"])
    properties["alternatives"].update({"minItems": 2, "maxItems": MAX_OPTIONS})
    properties["rationale"].update({"minItems": 1, "maxItems": MAX_PREFERENCES})
    option = cast(dict[str, JSONValue], properties["alternatives"]["items"])
    fields = cast(dict[str, dict[str, JSONValue]], option["properties"])
    fields["option_id"]["pattern"] = "^" + OPTION_ID_PATTERN + "$"
    fields["temperatures_c"].update({"minItems": 1, "maxItems": 2})
    return schema


class DecisionSchemaError(ValueError):
    """Only a fixed code and value-free structural report cross this boundary."""

    def __init__(self, check: DecisionSchemaCheck):
        first = check["issues"][0]
        self.code = first["section"].upper() + "_" + first["code"]
        self.check = check
        super().__init__(self.code)


def inspect_decision_schema(value: object) -> DecisionSchemaCheck:
    """Flow: Receives untrusted final object from run_investigation(); returns bounded value-free DecisionSchemaCheck for parse_decision_selection() and one correction decision."""
    issues: list[DecisionSchemaIssue] = []
    omitted = 0
    reported: set[tuple[str, int | None, str, str]] = set()

    def add(
        section: Literal["decision", "option", "preference"],
        index: int | None,
        field: str,
        code: SchemaIssueCode,
        expected: str,
        received: object = None,
        missing: bool = False,
    ) -> None:
        nonlocal omitted
        key = (section, index, field, code)
        if key in reported:
            return
        reported.add(key)
        issue: DecisionSchemaIssue = {
            "section": section,
            "option_index": index if section == "option" else None,
            "preference_index": index if section == "preference" else None,
            "field": field,
            "code": code,
            "expected": expected,
            "received_type": None
            if missing
            else (
                "null"
                if received is None
                else "boolean"
                if type(received) is bool
                else "string"
                if isinstance(received, str)
                else "array"
                if isinstance(received, list)
                else "object"
                if isinstance(received, dict)
                else "number"
                if type(received) in (int, float)
                else "unsupported_type"
            ),
        }
        if len(issues) < MAX_SCHEMA_ISSUES:
            issues.append(issue)
        else:
            omitted += 1

    def walk(
        item: object,
        schema: dict[str, JSONValue],
        section: Literal["decision", "option", "preference"],
        index: int | None,
        field: str,
    ) -> None:
        kind = schema["type"]
        valid_type = (
            isinstance(item, dict)
            if kind == "object"
            else isinstance(item, list)
            if kind == "array"
            else isinstance(item, str)
            if kind == "string"
            else type(item) in (int, float)
        )
        if not valid_type:
            add(section, index, field, "WRONG_TYPE", str(kind), item)
            return
        if kind == "object":
            obj = cast(dict[str, object], item)
            properties = cast(dict[str, dict[str, JSONValue]], schema["properties"])
            for key, child in properties.items():
                if key not in obj:
                    add(section, index, key, "MISSING_FIELD", str(child["type"]), missing=True)
                else:
                    walk(obj[key], child, section, index, key)
            if obj.keys() - properties.keys():
                # Never echo an unknown key: it can itself contain a sequence or credential.
                add(
                    section,
                    index,
                    "$additional_fields",
                    "UNEXPECTED_FIELD",
                    "only declared fields",
                    obj,
                )
        elif kind == "array":
            array = cast(list[object], item)
            lower, upper = (
                int(cast(int, schema.get("minItems", 0))),
                int(cast(int, schema["maxItems"])),
            )
            if not lower <= len(array) <= upper:
                add(section, index, field, "INVALID_COUNT", f"{lower}..{upper} items", array)
            for position, child_item in enumerate(array[:upper]):
                child_section, child_index = section, index
                if field == "alternatives":
                    child_section, child_index = "option", position
                elif field == "rationale":
                    child_section, child_index = "preference", position
                walk(
                    child_item,
                    cast(dict[str, JSONValue], schema["items"]),
                    child_section,
                    child_index,
                    "$object" if field in ("alternatives", "rationale") else field,
                )
        elif kind == "string":
            if "enum" in schema and item not in cast(list[str], schema["enum"]):
                add(section, index, field, "UNSUPPORTED_ENUM", json.dumps(schema["enum"]), item)
            if "pattern" in schema and re.fullmatch(OPTION_ID_PATTERN, cast(str, item)) is None:
                add(section, index, field, "INVALID_IDENTIFIER", OPTION_ID_PATTERN, item)
        else:
            try:
                finite = math.isfinite(cast(float, item))
            except OverflowError:
                finite = False
            if not finite:
                add(section, index, field, "NON_FINITE_NUMBER", "finite number", item)

    walk(value, decision_json_schema(), "decision", None, "$object")
    recoverable = (
        bool(issues)
        and omitted == 0
        and all(
            i["code"] not in ("UNEXPECTED_FIELD", "NON_FINITE_NUMBER")
            and (i["code"] != "UNSUPPORTED_ENUM" or i["field"] in ("design", "action"))
            for i in issues
        )
    )
    return {"issues": issues, "omitted_issue_count": omitted, "recoverable": recoverable}


def parse_decision_selection(value: object) -> DecisionSelection:
    """Flow: Receives untrusted final object from validate_experimental_decision(); returns structurally checked DecisionSelection for its unchanged scientific validators or raises DecisionSchemaError."""
    check = inspect_decision_schema(value)
    if check["issues"]:
        raise DecisionSchemaError(check)
    return cast(DecisionSelection, value)


def empty_contract_diagnostics() -> DecisionContractDiagnostics:
    """Flow: Receives a new run_investigation() boundary; returns empty DecisionContractDiagnostics for safe per-turn correction accounting, never provider content."""
    return {
        "original_schema_valid": None,
        "original_accepted": False,
        "initial_issues": [],
        "initial_omitted_issue_count": 0,
        "correction_attempts": 0,
        "correction_status": "NOT_REQUESTED",
        "corrected_accepted": False,
        "final_schema_valid": None,
        "final_issues": [],
        "final_omitted_issue_count": 0,
    }


def decision_contract_message(temperatures: list[float]) -> str:
    """Flow: Receives validated equipment list[float] from run_investigation(); returns schema/example str for its model instructions, never a grounded experimental recommendation."""
    # Illustrative labels/IDs cannot pass grounding. Only the numeric example must fit the outbound equipment guard.
    options = []
    for identifier, design, names, varied in (
        (
            "P1",
            "MATCHED_ENZYME_COMPARISON",
            ["Illustrative enzyme alpha", "Illustrative enzyme beta"],
            ["enzyme identity"],
        ),
        ("P2", "MEASUREMENT_QUALIFICATION", ["Illustrative enzyme alpha"], ["control condition"]),
    ):
        options.append(
            {
                "option_id": identifier,
                "design": design,
                "enzymes": names,
                "temperatures_c": temperatures[:1],
                "ph": 6.0,
                "pet_form": "CRYSTALLINE_POWDER",
                "controlled_variables": [v for v in CONTROLS if v not in varied],
                "varied_variables": varied,
                "controls": list(get_args(Control)),
                "evidence_ids": ["illustrative_source_id"],
            }
        )
    example = {
        "action": "EXPERIMENTAL_DECISION",
        "alternatives": options,
        "ranking": ["P1", "P2"],
        "rationale": [
            {
                "preferred": "P1",
                "over": "P2",
                "criterion": "DIRECT_ENZYME_DECISION",
                "evidence_ids": ["illustrative_source_id"],
            }
        ],
        "fact_ids": ["illustrative_source_id"],
    }
    return (
        "\nExact structural contract generated from Python DecisionSelection types: "
        + json.dumps(
            {
                "schema": decision_json_schema(),
                "illustrative_structural_example_not_evidence_or_recommendation": example,
                "example_rule": "Replace illustrative labels and IDs with executed source facts; choose your own alternatives, levels and priorities. Example pH/designs are not recommendations or source facts. All scientific validators still apply.",
            },
            allow_nan=False,
        )
    )


def decision_correction_message(check: DecisionSchemaCheck, required_ids: list[str]) -> str:
    """Flow: Receives value-free DecisionSchemaCheck and executed canonical list[str] from run_investigation(); returns one corrective system str for the same authenticated evidence, not a replacement proposal."""
    return (
        "One explicit output-schema correction is permitted within the remaining existing budget. Return final JSON only, no tool calls. Correct the structure yourself; do not change source facts, invent data, infer a winner or copy illustrative evidence. Every scientific validator still applies. "
        + json.dumps(
            {
                "structural_errors": check,
                "expected_structure": decision_json_schema(),
                "required_fact_ids": required_ids,
                "automatically_retained_kinds": ["LIMITATION", "UNANSWERED"],
            },
            allow_nan=False,
        )
    )
