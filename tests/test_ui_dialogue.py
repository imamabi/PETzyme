"""AppTest and worker transport checks; no live inference or real credential lookup."""

import io
import json
import os
import sys
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest
from test_dialogue import AMBIGUOUS, QUESTION, DialogueTransport, run_turn

from petzyme import ui_live

ROOT = Path(__file__).resolve().parents[1]


def test_actual_ui_callbacks_keep_only_context_and_numeric_cost(monkeypatch):
    calls = []

    def fake(question, **kwargs):
        assert kwargs["authorized"] and kwargs["completion_tokens"] == 10000
        assert 0 < kwargs["cost_ceiling_usd"] <= 10
        payload = kwargs["dialogue_input"]
        calls.append(1)
        scenario = (
            DialogueTransport(["ESM050", "LCCMC"])
            if payload["context"] is None
            else DialogueTransport(sequence=True)
        )
        return run_turn(payload, scenario)

    monkeypatch.setattr("petzyme.ui_dialogue.run_keychain_investigation", fake)
    at = AppTest.from_file(ROOT / "tests/legacy_streamlit_app.py", default_timeout=30).run()
    at.sidebar.radio[0].set_value("Live Muse dialogue").run()
    at.text_area(key="dialogue_text").input(QUESTION)
    at.checkbox(key="dialogue_authorize_0").check()
    at.button(key="dialogue_send").click().run()
    assert not at.exception and len(calls) == 1
    assert at.session_state["live_dialogue_context"]["confirmed_labels"] == ["ESM050", None]
    assert at.checkbox(key="dialogue_new").value is False
    assert at.text_area(key="dialogue_text").value == ""
    at.text_area(key="dialogue_text").input("Here is the sequence:\n" + AMBIGUOUS)
    at.checkbox(key="dialogue_authorize_1").check()
    at.button(key="dialogue_send").click().run()
    assert not at.exception and len(calls) == 2
    assert any("AMBIGUOUS_RESIDUES" in t.value and "30" in t.value for t in at.text)
    assert AMBIGUOUS not in str(at.session_state)
    from streamlit.runtime.state.session_state import SessionState

    internal = at.session_state
    while not isinstance(internal, SessionState):
        internal = object.__getattribute__(internal, "_state")
    assert AMBIGUOUS not in repr(internal._old_state) and AMBIGUOUS not in str(
        internal.get_widget_states()
    )
    at.run()
    assert len(calls) == 2
    assert not any("Local triage:" in t.value for t in at.text)
    assert "view" not in at.session_state
    spent = at.session_state["dialogue_spent"]
    at.button(key="dialogue_reset").click().run()
    assert (
        "live_dialogue_context" not in at.session_state
        and at.session_state["dialogue_spent"] == spent
    )
    at.sidebar.radio[0].set_value("Scientific clarification").run()
    assert "dialogue_text" not in at.session_state


def test_worker_with_synthetic_credential_and_actual_mock_transport(monkeypatch, capsys):
    from petzyme.meta import MetaModelClient

    scenario = DialogueTransport(["ESM050", "LCCMC"])
    backend = MetaModelClient(
        transport=httpx.MockTransport(scenario.respond),
        completion_tokens=10000,
        max_estimated_cost_usd=10,
    )
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-worker-only")
    monkeypatch.setattr("petzyme.meta.MetaModelClient", lambda **kwargs: backend)
    monkeypatch.setattr(sys, "argv", ["ui_live", "--live-worker"])
    payload = {
        "question": "Local identity dialogue",
        "completion_tokens": 10000,
        "dialogue_input": {"context": None, "text": QUESTION, "input_kind": "ANSWER"},
    }
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode())))
    assert ui_live.worker_main() == 0
    output = capsys.readouterr().out
    assert "synthetic-worker-only" not in output
    result = json.loads(output.splitlines()[-1])["result"]["result"]
    assert result["dialogue"]["outcome"] == "ASK_CLARIFICATION"
    assert len(result["provider_telemetry"]) == 2


def test_parent_launcher_worker_dialogue_boundary(tmp_path, monkeypatch):
    for package in ("scripts", "petzyme"):
        directory = tmp_path / package
        directory.mkdir()
        (directory / "__init__.py").write_text(
            "__path__.append(" + repr(str(ROOT / package)) + ")\n"
        )
    common = """import importlib.util,socket,sys,os
from pathlib import Path
ROOT=Path(%r)
def blocked(*a,**k):raise AssertionError('Offline child must never connect')
socket.socket.connect=blocked
""" % str(ROOT)
    (tmp_path / "scripts/launch_meta.py").write_text(
        common
        + """
import subprocess
spec=importlib.util.spec_from_file_location('scripts._real_launch_meta',ROOT/'scripts/launch_meta.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert os.environ.get('MODEL_API_KEY') == 'synthetic-parent-must-not-forward'
module._lookup=lambda password:subprocess.CompletedProcess([],0,b'synthetic-child-only\\n')
raise SystemExit(module.main())
"""
    )
    (tmp_path / "petzyme/ui_live.py").write_text(
        common
        + """
import httpx
from tests.test_dialogue import DialogueTransport
from petzyme import ui_adapter,meta
assert os.environ.get('MODEL_API_KEY')=='synthetic-child-only'
scenario=DialogueTransport(['ESM050','LCCMC'])
Original=meta.MetaModelClient
meta.MetaModelClient=lambda **kwargs:Original(transport=httpx.MockTransport(scenario.respond),completion_tokens=kwargs['completion_tokens'],max_estimated_cost_usd=kwargs.get('max_estimated_cost_usd'))
spec=importlib.util.spec_from_file_location('petzyme._real_ui_live',ROOT/'petzyme/ui_live.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
raise SystemExit(module.worker_main())
"""
    )
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + str(ROOT) + os.pathsep + str(ROOT / "tests")
    )
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-parent-must-not-forward")
    popen = ui_live.subprocess.Popen
    commands = []

    def observe(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["MODEL_API_KEY"] == "synthetic-parent-must-not-forward"
        return popen(command, **kwargs)

    monkeypatch.setattr(ui_live.subprocess, "Popen", observe)
    from petzyme.runtime_config import RuntimeLimits

    limits = RuntimeLimits(turn_models=3, turn_tools=4, session_models=12, session_tools=16)
    result = ui_live.run_keychain_investigation(
        "Local identity dialogue",
        authorized=True,
        completion_tokens=10000,
        dialogue_input={"context": None, "text": QUESTION, "input_kind": "ANSWER"},
        cost_ceiling_usd=10,
        runtime_limits=limits,
    )
    assert (
        result["result"] is not None
        and result["result"]["dialogue"]["outcome"] == "ASK_CLARIFICATION"
    )
    assert len(commands) == 1 and commands[0][1:5] == [
        "-m",
        "scripts.launch_meta",
        "--authorize-paid-meta",
        "--",
    ]
    assert result["result"]["limits"] == limits.values()
    assert result["result"]["dialogue"]["context"]["usage"]["actual_http_requests"] == 2
    assert "synthetic-" not in json.dumps(result)


def test_incomplete_dialogue_panel_is_honest_and_does_not_label_mock_http_as_real(monkeypatch):
    def fake(question, **kwargs):
        return run_turn(kwargs["dialogue_input"], DialogueTransport(["ESM050"], bad="ids"))

    monkeypatch.setattr("petzyme.ui_dialogue.run_keychain_investigation", fake)
    at = AppTest.from_file(ROOT / "tests/legacy_streamlit_app.py", default_timeout=30).run()
    at.sidebar.radio[0].set_value("Live Muse dialogue").run()
    at.text_area(key="dialogue_text").input(QUESTION)
    at.checkbox(key="dialogue_authorize_0").check()
    at.button(key="dialogue_send").click().run()
    assert not at.exception
    assert any("Live Muse dialogue incomplete" in error.value for error in at.error)
    assert not any("Could you provide" in info.value for info in at.info)
    assert not any("Actual Meta HTTP" in caption.value for caption in at.caption)
    assert any("Mock transport HTTP attempts" in caption.value for caption in at.caption)
    assert "view" not in at.session_state


def test_legacy_context_requires_explicit_reset_without_faking_usage_or_paid_calls(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Reset must not request paid inference")

    monkeypatch.setattr("petzyme.ui_dialogue.run_keychain_investigation", blocked)
    at = AppTest.from_file(ROOT / "tests/legacy_streamlit_app.py", default_timeout=30).run()
    at.sidebar.radio[0].set_value("Live Muse dialogue").run()
    legacy = {"requested_labels": ["ESM050", "LCCMC"], "model_attempts": 2}
    at.session_state["live_dialogue_context"] = legacy
    at.session_state["dialogue_spent"] = 0.02
    at.run()
    assert not at.exception and any("older or invalid contract" in w.value for w in at.warning)
    assert at.session_state["live_dialogue_context"] == legacy
    at.button(key="dialogue_reset").click().run()
    assert "live_dialogue_context" not in at.session_state
    assert at.session_state["dialogue_spent"] == 0.02
