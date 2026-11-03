"""One authorized credential-bearing child per live investigation."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import selectors
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, cast

from . import meta
from .agent_request import interpret_request
from .agent_types import ToolUpdate
from .clarification import parse_clarification_context
from .contracts import ExperimentalWindow, ToolResult
from .dialogue import prepare_dialogue_turn
from .dialogue_types import DialogueInput, LiveInvestigationInput
from .experimental_window import valid_window
from .meta import MAX_COMPLETION_TOKENS, valid_completion_allowance
from .results import outcome
from .runtime_config import MAX_SESSION_COST_USD, RuntimeLimits, parse_runtime_limits
from .scientific_scope import ScientificScopeError, validate_direct_dialogue_scope
from .ui_adapter import ROOT, ResearchView, investigate_with_backend, runtime_identity

WORKER_TIMEOUT_SECONDS = 1200


def general_research_question(payload: DialogueInput) -> str | None:
    """Flow: Receives external DialogueInput from run_keychain_investigation(); returns a privacy-validated research question str or None for its existing structured dialogue worker, without fabricating retained context."""
    if (
        not isinstance(payload, dict)
        or payload.get("context") is not None
        or payload.get("input_kind") != "ANSWER"
        or payload.get("decision_support", False)
        or not {"context", "text", "input_kind"}
        <= payload.keys()
        <= {"context", "text", "input_kind", "decision_support"}
    ):
        return None
    text = payload["text"]
    interpret_request(text)
    validate_direct_dialogue_scope(text)
    try:
        parse_clarification_context(text)
    except ValueError:
        # An incomplete enzyme-choice request must retain its constraint/identity
        # validation rather than silently becoming an unrelated investigation.
        if not re.search(r"\b(?:should\s+I\s+use|choose\s+between)\b", text, re.I):
            return text
    return None


def run_keychain_investigation(
    question: str,
    *,
    authorized: bool = False,
    on_update: Callable[[ToolUpdate], None] | None = None,
    completion_tokens: int = MAX_COMPLETION_TOKENS,
    dialogue_input: DialogueInput | None = None,
    cost_ceiling_usd: float | None = None,
    runtime_limits: RuntimeLimits | None = None,
    automatic_runtime: bool = False,
    cohort_origin: str | None = None,
    cohort_filter: str | None = None,
    cohort_compare_filters: tuple[str, str] | None = None,
    cohort_compare_scopes: tuple[bool, bool] | None = None,
    cohort_pazy_only: bool = False,
    cohort_fingerprint: str | None = None,
    cohort_window: ExperimentalWindow | None = None,
    context_turns: list[str] | None = None,
) -> ToolResult[ResearchView]:
    """Run one live investigation in an authorized child process.

    Purpose: Streamlit spawns the existing credential launcher as a subprocess,
        sends the bounded
        question JSON over stdin, and reads back public ToolUpdate /
        result JSON lines. Each submission needs a fresh explicit
        authorization; without it, nothing runs even if a key exists.
    Flow: Receives external question str and explicit per-investigation authorization from Streamlit main(); returns ToolResult[ResearchView] from worker_main() via the existing Keychain launcher."""
    if authorized is not True:
        return outcome(
            "ERROR",
            None,
            [],
            [],
            None,
            "Authorize live Meta inference for this investigation, then press Investigate.",
        )
    if type(automatic_runtime) is not bool:
        return outcome("ERROR", None, [], [], None, "Invalid runtime policy")
    if not valid_completion_allowance(completion_tokens):
        return outcome(
            "ERROR", None, [], [], None, "Invalid completion allowance; no child was launched."
        )
    if runtime_limits is not None and not isinstance(runtime_limits, RuntimeLimits):
        return outcome("ERROR", None, [], [], None, "Invalid typed runtime configuration")
    if context_turns is not None and not isinstance(context_turns, list):
        return outcome("ERROR", None, [], [], None, "Invalid separate research context")
    if cost_ceiling_usd is not None:
        if (
            type(cost_ceiling_usd) not in (int, float)
            or not math.isfinite(cost_ceiling_usd)
            or not 0 < cost_ceiling_usd <= MAX_SESSION_COST_USD
        ):
            return outcome("ERROR", None, [], [], None, "Invalid temporary spending ceiling")
    try:
        general = general_research_question(dialogue_input) if dialogue_input is not None else None
        if general is not None:
            question, dialogue_input = general, None
        if dialogue_input is None:
            interpret_request(question)
        else:
            _, context, _ = prepare_dialogue_turn(dialogue_input)
            limits = runtime_limits or RuntimeLimits()
            if not automatic_runtime and (
                context["model_attempts"] >= limits.session_models
                or context["tool_attempts"] >= limits.session_tools
            ):
                return outcome(
                    "ERROR",
                    None,
                    [],
                    [],
                    None,
                    "Session attempt limit exhausted; no child launched",
                )
            if not automatic_runtime and (
                context["usage"]["reserved_cost_usd"]
                >= (runtime_limits or RuntimeLimits()).session_cost_usd
            ):
                return outcome(
                    "ERROR", None, [], [], None, "Session cost ceiling exhausted; no child launched"
                )
    except ScientificScopeError as error:
        return outcome("ERROR", None, [], [], None, str(error))
    except ValueError:
        return outcome(
            "ERROR",
            None,
            [],
            [],
            None,
            "Supply a supported research question without sequence or credential content.",
        )
    command = [
        sys.executable,
        "-m",
        "scripts.launch_meta",
        "--authorize-paid-meta",
        "--",
        sys.executable,
        "-m",
        "petzyme.ui_live",
        "--live-worker",
    ]
    # The fixed local launcher receives the process credential when supplied;
    # it selects Keychain first on macOS and forwards it only to the worker.
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1"
    )
    if cost_ceiling_usd is not None:
        environment["ZYMECYCLE_META_COST_CEILING_USD"] = str(cost_ceiling_usd)
    limits = runtime_limits or RuntimeLimits()
    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=sys.platform != "win32",
        )
        if process.stdin is None or process.stdout is None:
            raise ValueError("Worker pipes are unavailable")
        payload: LiveInvestigationInput = {
            "question": question,
            "completion_tokens": completion_tokens,
        }
        if automatic_runtime:
            payload["automatic_runtime"] = True
        if dialogue_input is not None:
            payload["question"] = "Local identity dialogue"
            payload["dialogue_input"] = dialogue_input
        if runtime_limits is not None:
            payload["runtime_limits"] = limits.values()
        if context_turns:
            if (not isinstance(context_turns, list) or dialogue_input is not None or cohort_origin is not None
                    or any(not isinstance(turn, str) for turn in context_turns)):
                raise ValueError("Invalid separate research context")
            for turn in context_turns:
                interpret_request(turn)
            payload["context_turns"] = context_turns
        if cohort_origin is not None:
            if dialogue_input is not None:
                raise ValueError("Cohort continuation cannot use a dialogue payload")
            interpret_request(cohort_origin)
            payload["cohort_origin"] = cohort_origin
            payload["cohort_pazy_only"] = cohort_pazy_only
            if cohort_fingerprint is None:
                raise ValueError("Cohort continuation requires a code fingerprint")
            payload["cohort_fingerprint"] = cohort_fingerprint
            if cohort_window is not None:
                if not valid_window(cohort_window):
                    raise ValueError("Invalid laboratory planning constraints")
                payload["cohort_window"] = cohort_window
            if cohort_filter is not None:
                interpret_request(cohort_filter)
                payload["cohort_filter"] = cohort_filter
            if cohort_compare_filters is not None:
                if len(cohort_compare_filters) != 2:
                    raise ValueError("Two historical filters are required")
                for item in cohort_compare_filters:
                    interpret_request(item)
                payload["cohort_compare_filters"] = list(cohort_compare_filters)
                if cohort_compare_scopes is not None:
                    if len(cohort_compare_scopes) != 2 or any(type(value) is not bool for value in cohort_compare_scopes):
                        raise ValueError("Invalid historical comparison source scopes")
                    payload["cohort_compare_scopes"] = list(cohort_compare_scopes)
        process.stdin.write(json.dumps(payload).encode() + b"\n")
        process.stdin.close()
        deadline, pending, received, messages = (
            time.monotonic()
            + max(WORKER_TIMEOUT_SECONDS, limits.turn_models * meta.REQUEST_TIMEOUT_SECONDS + 60),
            b"",
            0,
            0,
        )
        result = None
        with selectors.DefaultSelector() as selector:
            chunks: queue.Queue[bytes] = queue.Queue(maxsize=2)
            if sys.platform == "win32":
                def read_pipe() -> None:
                    try:
                        while True:
                            data = os.read(process.stdout.fileno(), 8192)
                            chunks.put(data)
                            if not data:
                                return
                    except OSError:
                        chunks.put(b"")
                threading.Thread(target=read_pipe, daemon=True).start()
            else:
                selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if time.monotonic() > deadline:
                    raise ValueError("Worker deadline reached")
                if sys.platform == "win32":
                    try:
                        chunk = chunks.get(timeout=0.25)
                    except queue.Empty:
                        continue
                else:
                    if not selector.select(timeout=0.25):
                        continue
                    chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                received += len(chunk)
                if (not automatic_runtime and received > 1_048_576) or len(pending) > meta.MAX_AUTOMATIC_RESPONSE_BYTES:
                    raise ValueError("Worker result bound exceeded")
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    messages += 1
                    if not automatic_runtime and messages > limits.turn_tools + 1:
                        raise ValueError("Worker message budget exceeded")
                    message = json.loads(line)
                    if set(message) == {"update"} and result is None:
                        if on_update is not None:
                            on_update(message["update"])
                    elif set(message) == {"result"} and result is None:
                        result = message["result"]
                    else:
                        raise ValueError("Invalid worker protocol")
        code = process.wait(timeout=5)
        valid_result = (
            isinstance(result, dict)
            and {"status", "result", "provenance", "limitations", "evidence_category", "message"}
            <= result.keys()
            and result["status"] in ("SUCCESS", "PARTIAL", "NO_DATA", "ERROR")
            and (
                result["result"] is None
                or isinstance(result["result"], dict)
                and ResearchView.__required_keys__ <= result["result"].keys()
            )
        )
        if code == 0 and valid_result and not pending:
            return cast(ToolResult[ResearchView], result)
        return outcome(
            "ERROR",
            None,
            [],
            [],
            None,
            "Live worker startup or public-result protocol failed. This does not establish a credential failure. Check worker execution and login Keychain access. No retry was made.",
        )
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, subprocess.TimeoutExpired):
        return outcome(
            "ERROR",
            None,
            [],
            [],
            None,
            "Live process stopped without a valid public result. No retry was made.",
        )
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    if sys.platform == "win32":
                        system_root = os.environ.get("SystemRoot", r"C:\Windows")
                        taskkill = os.path.join(system_root, "System32", "taskkill.exe")
                        subprocess.run([taskkill, "/PID", str(process.pid), "/T", "/F"],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=5, check=False)
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        if sys.platform == "win32":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()


def worker_main() -> int:
    """Child entry point: run the investigation with the Keychain-provided key.

    Purpose: reads the bounded question JSON from stdin, executes via
        investigate_with_backend with a live MetaModelClient, and emits
        only public JSON lines on stdout. The key lives solely in this
        child's environment and is never printed.
    Flow: Receives bounded question JSON on external stdin from run_keychain_investigation(); emits public ToolUpdate/result JSON lines from investigate_with_backend() and returns final exit status int."""
    if sys.argv[1:] != ["--live-worker"] or not os.environ.get("MODEL_API_KEY"):
        return 1
    try:
        raw = sys.stdin.buffer.readline()
        request = json.loads(raw)
        if not isinstance(request, dict) or not {"question"} <= request.keys() <= {
            "question",
            "completion_tokens",
            "dialogue_input",
            "runtime_limits",
            "cohort_origin",
            "cohort_filter",
            "cohort_compare_filters",
            "cohort_compare_scopes",
            "automatic_runtime",
            "cohort_pazy_only",
            "cohort_fingerprint",
            "cohort_window",
            "context_turns",
        }:
            return 1
        limits = parse_runtime_limits(request.get("runtime_limits"))
        automatic_runtime = request.get("automatic_runtime", False)
        if type(automatic_runtime) is not bool:
            return 1
        dialogue = {}
        previous_cost = 0.0
        question = request["question"]
        if "cohort_origin" in request:
            if request.get("cohort_fingerprint") != runtime_identity()["code_fingerprint"]:
                print(json.dumps({"result": outcome("ERROR", None, [], [], None,
                      "Application and worker code differ; restart the app and start a new discovery.")}), flush=True)
                return 0
        if "dialogue_input" in request:
            question, context, sequence = prepare_dialogue_turn(request["dialogue_input"])
            previous_cost = context["usage"]["reserved_cost_usd"]
            dialogue = {"dialogue_input": request["dialogue_input"]}
        else:
            interpret_request(question)
        allowance = request.get("completion_tokens", MAX_COMPLETION_TOKENS)
        if not valid_completion_allowance(allowance):
            return 1
        ceiling = os.environ.get("ZYMECYCLE_META_COST_CEILING_USD")
        max_cost = None if automatic_runtime else min(
            float(ceiling) if ceiling is not None else limits.session_cost_usd,
            limits.session_cost_usd - previous_cost,
        )
        if max_cost is not None and max_cost <= 0:
            print(
                json.dumps(
                    {
                        "result": outcome(
                            "ERROR",
                            None,
                            [],
                            [],
                            None,
                            "Session cost ceiling exhausted; no model request made",
                        )
                    }
                ),
                flush=True,
            )
            return 0

        def emit(update: ToolUpdate) -> None:
            print(json.dumps({"update": update}, allow_nan=False), flush=True)

        result = investigate_with_backend(
            question,
            meta.MetaModelClient(
                live_authorized=True, completion_tokens=allowance, max_estimated_cost_usd=max_cost,
                automatic_runtime=automatic_runtime,
            ),
            on_update=emit,
            runtime_limits=limits,
            automatic_runtime=automatic_runtime,
            cohort_origin=request.get("cohort_origin"),
            cohort_filter=request.get("cohort_filter"),
            cohort_compare_filters=request.get("cohort_compare_filters"),
            cohort_compare_scopes=request.get("cohort_compare_scopes"),
            cohort_pazy_only=request.get("cohort_pazy_only", False),
            cohort_window=request.get("cohort_window"),
            context_turns=request.get("context_turns"),
            **dialogue,
        )
        print(json.dumps({"result": result}, allow_nan=False), flush=True)
        return 0
    except Exception:
        # No provider bodies, credentials, source records or exception diagnostics are emitted.
        return 1


if __name__ == "__main__":
    raise SystemExit(worker_main())
