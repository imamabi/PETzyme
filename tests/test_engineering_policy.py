"""Authoritative policy and finite configuration checks; no inference or source writes."""

import pytest

from petzyme.privacy import TOOL_PRIVACY, outbound_safe, tool_privacy
from petzyme.runtime_config import RuntimeLimits, parse_runtime_limits
from petzyme.tools import TOOL_FUNCTIONS


def test_every_registered_tool_has_one_policy_and_unknown_is_conservative():
    assert set(TOOL_PRIVACY) == set(TOOL_FUNCTIONS)
    assert tool_privacy("unknown").persistence == "FORBIDDEN_PAZY_DERIVED"
    assert tool_privacy("triage_protein").persistence == "FORBIDDEN_PAZY_DERIVED"
    assert tool_privacy("advise_next_experiment").communication_allowed
    assert not tool_privacy("get_pazy_metadata").communication_allowed


@pytest.mark.parametrize(
    "value",
    [{"sequence_sha256": "abc"}, {"embedding": [1, 2]}, {"rows": []}, "LLM|secret", "A" * 40],
)
def test_shared_outbound_privacy_rejects_forbidden_content(value):
    assert not outbound_safe(value)


def test_budgets_are_configurable_and_separate_turns_from_sessions():
    limits = RuntimeLimits(
        turn_models=8, turn_tools=12, session_models=40, session_tools=60, session_cost_usd=12
    )
    assert parse_runtime_limits(limits.values()) == limits
    assert RuntimeLimits().session_models > RuntimeLimits().turn_models


@pytest.mark.parametrize(
    "kwargs",
    [
        {"turn_models": 0},
        {"turn_tools": True},
        {"session_models": 129},
        {"session_cost_usd": float("inf")},
        {"session_models": 2},
    ],
)
def test_unbounded_or_incoherent_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RuntimeLimits(**kwargs)


def test_registry_execution_and_projection_use_the_same_authoritative_policy(monkeypatch):
    from petzyme.agent_evidence import project_evidence
    from petzyme.agent_execution import execute_tool_call
    from petzyme.agent_types import AgentState
    from petzyme.results import outcome

    error = outcome("ERROR", None, [], [], None, "Synthetic failure")
    monkeypatch.setattr("petzyme.agent_execution.invoke_tool", lambda *args, **kwargs: error)
    for index, name in enumerate(TOOL_FUNCTIONS):
        state = AgentState("unused")
        call = {"id": f"x{index}", "function": {"name": name, "arguments": "{}"}}
        catalog = [
            {
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            }
        ]
        event = execute_tool_call(call, state, catalog)
        _, facts = project_evidence(call["id"], name, error)
        expected = tool_privacy(name).persistence != "NREL_ONLY"
        assert event["ephemeral"] == expected
        assert all(f["ephemeral"] == expected for f in facts)


def test_short_private_sequence_is_blocked_even_in_the_legacy_question_path():
    from test_build10_acceptance import AMBIGUOUS

    from petzyme.agent_request import interpret_request
    from petzyme.privacy import question_contains_private_input

    assert question_contains_private_input("Here is the sequence: " + AMBIGUOUS)
    assert not outbound_safe("Here is the sequence: " + AMBIGUOUS, private_sequences=[AMBIGUOUS])
    with pytest.raises(ValueError):
        interpret_request("Here is the sequence: " + AMBIGUOUS)


def test_only_verified_equipment_arrays_can_cross_the_shared_boundary():
    assert outbound_safe(
        {"preserved_constraints": {"temperatures_c": [40, 60]}},
        approved_arrays={"temperatures_c": [40, 60]},
    )
    assert not outbound_safe(
        {"temperatures_c": [1, 2]}, approved_arrays={"temperatures_c": [40, 60]}
    )
    assert not outbound_safe({"embedding": [40, 60]}, approved_arrays={"temperatures_c": [40, 60]})


def test_import_graph_is_acyclic_including_function_local_imports():
    import ast
    from pathlib import Path

    graph = {p.stem: set() for p in Path("petzyme").glob("*.py")}
    for p in Path("petzyme").glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dependency = node.module.split(".")[0]
                if dependency in graph:
                    graph[p.stem].add(dependency)
    active = set()
    done = set()

    def visit(module):
        assert module not in active, "Import cycle: " + module
        if module in done:
            return
        active.add(module)
        for dependency in graph[module]:
            visit(dependency)
        active.remove(module)
        done.add(module)

    for module in graph:
        visit(module)
    for name in ("contracts", "runtime_config", "privacy", "dialogue_types"):
        assert not any(
            dep.startswith(("agent", "ui_")) or dep in ("dialogue", "dialogue_rules")
            for dep in graph[name]
        )
    assert "dialogue" not in graph["agent"] and "dialogue" not in graph["agent_execution"]
    assert "agent" in graph["dialogue"]


def test_agent_dialogue_boundary_imports_are_explicit_at_module_level():
    import ast
    from pathlib import Path

    for name in ("agent", "agent_execution", "dialogue", "dialogue_rules", "ui_adapter", "ui_live"):
        tree = ast.parse(Path("petzyme", name + ".py").read_text())
        for function in (
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            assert not any(
                isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(function)
            ), name


def test_source_chemical_terminology_is_not_reclassified_as_private_sequence():
    from petzyme.privacy import question_contains_private_input, safe_public_text

    chemical = "hydroxyethylterephthalate"
    assert safe_public_text(chemical) == chemical
    assert not question_contains_private_input("What does " + chemical + " mean here?")
