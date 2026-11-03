"""Small callable catalog and JSON dispatcher; no LLM, persistence or orchestration."""

from __future__ import annotations

import inspect
import json
import types
from typing import (
    Callable,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from .advisor import recommend_next_experiments
from .comparability import check_comparability
from .conditions import investigate_conditions
from .contracts import JSONValue, ToolResult
from .enzyme_discovery import discover_pet_enzymes
from .evidence import assess_evidence
from .gaps import identify_evidence_gaps
from .identity import compare_identity
from .metadata import get_associated_literature, get_pazy_metadata, search_pazy_pet
from .observations import (
    filter_observations,
    load_nrel,
    parse_nrel,
    select_nrel_protein,
    summarize_condition,
)
from .pet_evidence import investigate_pet_evidence
from .privacy import EPHEMERAL_TOOLS as EPHEMERAL_TOOLS
from .privacy import TOOL_PRIVACY, tool_privacy
from .protein_similarity import find_similar_nrel_proteins
from .protein_triage import triage_protein
from .research_advisor import advise_next_experiment
from .results import finite_number, outcome

TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {
    function.__name__: function
    for function in (
        load_nrel,
        parse_nrel,
        filter_observations,
        summarize_condition,
        select_nrel_protein,
        search_pazy_pet,
        get_pazy_metadata,
        get_associated_literature,
        compare_identity,
        assess_evidence,
        check_comparability,
        investigate_conditions,
        identify_evidence_gaps,
        find_similar_nrel_proteins,
        recommend_next_experiments,
        investigate_pet_evidence,
        advise_next_experiment,
        triage_protein,
        discover_pet_enzymes,
    )
}


def _schema(annotation: object) -> dict[str, JSONValue]:
    """Translate the supported input type vocabulary to JSON Schema without dependencies."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (NotRequired, Required):
        return _schema(args[0])
    if origin in (Union, types.UnionType):
        return {"anyOf": [_schema(arg) for arg in args]}
    if origin is Literal:
        return {"enum": list(args)}
    if is_typeddict(annotation):
        fields = get_type_hints(annotation, include_extras=True)
        return {
            "type": "object",
            "properties": {key: _schema(value) for key, value in fields.items()},
            "required": [name for name, value in fields.items()
                         if get_origin(value) is Required or
                         (get_origin(value) is not NotRequired and name in annotation.__required_keys__)],
            "additionalProperties": False,
        }
    if origin is list:
        return {"type": "array", "items": _schema(args[0])}
    primitives = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    if annotation in primitives:
        return {"type": primitives[annotation]}
    raise TypeError("Unsupported input type in scientific tool catalog")


def _parameters(function: Callable[..., ToolResult]) -> dict[str, JSONValue]:
    """Describe JSON arguments; dependency-injection clients are not tool arguments."""
    signature, hints = inspect.signature(function), get_type_hints(function)
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        if name == "client":
            continue
        properties[name] = _schema(hints[name])
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _valid(value: JSONValue, schema: dict[str, JSONValue]) -> bool:
    """Validate the catalog's small JSON Schema vocabulary without coercion."""
    if "anyOf" in schema:
        return any(_valid(value, candidate) for candidate in schema["anyOf"])
    if "enum" in schema:
        return value in schema["enum"]
    kind = schema["type"]
    if kind in ("integer", "number"):
        valid_type = type(value) is int if kind == "integer" else finite_number(value)
        return (
            valid_type
            and ("minimum" not in schema or value >= schema["minimum"])
            and ("maximum" not in schema or value <= schema["maximum"])
        )
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema["properties"]
        return set(schema["required"]) <= value.keys() <= properties.keys() and all(
            _valid(item, properties[key]) for key, item in value.items()
        )
    if kind == "array":
        return isinstance(value, list) and all(_valid(item, schema["items"]) for item in value)
    expected = {"string": str, "integer": int, "boolean": bool, "null": type(None)}[kind]
    return type(value) is expected


def validate_arguments(value: JSONValue, schema: dict[str, JSONValue]) -> bool:
    """Validate JSON arguments against the catalog's small schema vocabulary.

    Purpose: dependency-free strict check (anyOf/enum/typed
        objects/arrays with exact keys and no coercion) shared by the
        dispatcher and the agent's call validation.
    Returns:
        bool; True only when the value fully satisfies the schema.
    Flow: Receives JSON arguments and schema from select_tool_catalog() or invoke_tool(); returns bool for execute_tool_call() or invoke_tool()."""
    return _valid(value, schema)


def describe_tools() -> ToolResult[list[dict[str, JSONValue]]]:
    """Describe every registered tool: name, docstring, JSON schema, output type, persistence.

    Purpose: callable catalog for discovery and for select_tool_catalog.
        Refuses to run unless every registered tool has an explicit
        privacy policy, so a new tool without a persistence decision
        breaks loudly instead of defaulting.
    Returns:
        ToolResult[list[dict]] with one descriptor per tool ('name',
            'description', 'input_schema', 'output_type',
            'output_persistence'). The 'client' injection parameter is
            excluded from input schemas.
    Raises:
        RuntimeError: when catalog and privacy policies disagree.
    Example:
        >>> out = describe_tools()
        >>> (out['status'], len(out['result']))
        ('SUCCESS', 18)
    Flow: Receives external discovery requests with no arguments; returns ToolResult[list[dict[str, JSONValue]]] for callers of invoke_tool()."""
    if set(TOOL_FUNCTIONS) != set(TOOL_PRIVACY):
        raise RuntimeError("Every registered scientific tool requires an explicit privacy policy")
    descriptions: list[dict[str, JSONValue]] = []
    for name, function in TOOL_FUNCTIONS.items():
        descriptions.append(
            {
                "name": name,
                "description": inspect.getdoc(function),
                "input_schema": _parameters(function),
                "output_type": str(inspect.signature(function).return_annotation),
                "output_persistence": tool_privacy(name).persistence,
            }
        )
    return outcome("SUCCESS", descriptions, [], [], None)


def invoke_tool(name: str, arguments: dict[str, JSONValue]) -> ToolResult[JSONValue]:
    """Validate JSON arguments and invoke one registered tool (no orchestration).

    Purpose: the single JSON entry point to the science layer. Looks up
        the tool, validates arguments strictly, calls it, and verifies
        the result is JSON-serializable (in memory only — never printed
        or persisted). Unknown tools and contract violations return
        ERROR envelopes instead of raising.
    Args:
        name: registered tool name from describe_tools().
        arguments: JSON arguments matching the tool's input schema.
    Returns:
        ToolResult[JSONValue]: the callable's own output on success, or
            an ERROR envelope naming the contract failure.
    Example:
        >>> out = invoke_tool('load_nrel', {'csv_path': 'data/nrel/label_data.csv'})
        >>> (out['status'], out['result']['columns'][:2])
        ('SUCCESS', ['', 'sequence'])
    Flow: Receives tool name str from describe_tools() and external JSON arguments; returns ToolResult[JSONValue] as the selected callable's final output."""
    if not isinstance(name, str):
        return outcome("ERROR", None, [], [], None, "Tool name must be a string")
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return outcome("ERROR", None, [], [], None, "Unknown scientific tool")
    if not validate_arguments(arguments, _parameters(function)):
        return outcome(
            "ERROR", None, [], [], None, "Arguments do not match the tool input contract"
        )
    try:
        result = function(**arguments)
        # In-memory validation only. Never print, log or persist this serialized value.
        json.dumps(result, allow_nan=False)
    except (KeyError, TypeError, ValueError, OverflowError):
        return outcome(
            "ERROR", None, [], [], None, "Tool input or result did not satisfy its contract"
        )
    return result
