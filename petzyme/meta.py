"""Official Meta Model API Chat Completions adapter; live use is disabled by default.

Verified schema: https://dev.meta.ai/docs/api-reference/chat-completions/schemas
Tool protocol: https://dev.meta.ai/docs/tool-calling
"""
from __future__ import annotations

import json
import math
import os
import time
import httpx
from .agent_types import CALL_ID, ReasoningError
from .meta_telemetry import RequestTelemetry, projected_token_cost, record_usage

ENDPOINT = 'https://api.meta.ai/v1/chat/completions'
MODEL = 'muse-spark-1.3'
MAX_BYTES = 262_144
# A single provider response remains memory-bounded against malformed streams;
# this is not a model-request or session-token quota.
MAX_AUTOMATIC_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_COMPLETION_TOKENS = 2048
COMPLETION_ALLOWANCES = (10000, 20000, 25000, 50000)
REQUEST_TIMEOUT_SECONDS = 180


def valid_completion_allowance(value: object) -> bool:
    """Accept only the validated completion-token allowances (2048 default; 10k/20k/25k/50k).

    Purpose: prevents arbitrary token budgets; each allowance was
        validated against the provider envelope and context-size guard.
    Flow: Receives external per-investigation allowance object; returns bool for MetaModelClient.__init__() and run_keychain_investigation()."""
    return type(value) is int and value in (MAX_COMPLETION_TOKENS, *COMPLETION_ALLOWANCES)


class ModelError(ReasoningError):
    """Sanitized provider/configuration failure; never includes keys or response bodies."""


class CompletionLimitError(ModelError):
    """Length-limited completion, with a safe provider-neutral reason code."""
    code = 'COMPLETION_LENGTH_LIMIT'


class ContextLimitError(ModelError):
    """Local serialized-context bound, before any provider request is sent."""
    code = 'REQUEST_SIZE_LIMIT'


class MetaModelClient:
    """Official Meta chat-completions adapter; live use requires explicit authorization.

    Purpose: provider behind the ReasoningBackend protocol. Reads
        MODEL_API_KEY from the process environment only in memory, sends
        bounded non-streamed requests with store=false, records numeric
        telemetry per request, and enforces an optional pre-request
        spending ceiling. No retries, no logging, no persistent history;
        tests inject a MockTransport instead of going live."""
    def __init__(self, *, live_authorized: bool = False,
                 transport: httpx.MockTransport | None = None,
                 completion_tokens: int = MAX_COMPLETION_TOKENS,
                 max_estimated_cost_usd: float | None = None,
                 automatic_runtime: bool = False) -> None:
        """Configure authorization, completion allowance, and spending ceiling.

        Args:
            live_authorized: True only after explicit paid-call approval.
            transport: test-only httpx.MockTransport (nothing else may
                bypass the authorization gate).
            completion_tokens: one validated allowance (default 2048).
            max_estimated_cost_usd: optional finite positive ceiling;
                requests that would exceed it are blocked unsent.
        Raises:
            ValueError: on a non-mock transport, bad allowance, or bad ceiling.
        Flow: Receives external authorization bool or test MockTransport; returns None after initializing complete() request controls."""
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError('Only httpx.MockTransport may bypass the live authorization gate')
        if not valid_completion_allowance(completion_tokens):
            raise ValueError('Select one of the four validated completion allowances')
        if type(automatic_runtime) is not bool:
            raise ValueError('Invalid runtime policy')
        if max_estimated_cost_usd is not None and (type(max_estimated_cost_usd) not in (int,float)
                or not math.isfinite(max_estimated_cost_usd) or max_estimated_cost_usd <= 0):
            raise ValueError('The optional estimated spending ceiling must be finite and positive')
        self.max_estimated_cost_usd = max_estimated_cost_usd
        self.automatic_runtime = automatic_runtime
        self.budget_blocked = False
        self.completion_tokens = completion_tokens
        self.telemetry: list[RequestTelemetry] = []
        self._live_authorized = live_authorized is True
        self._transport = transport
        self.mode = 'MOCK_META' if transport is not None else 'LIVE_META_DISABLED' if not live_authorized else 'LIVE_META'

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """Send one bounded chat-completion request; validate and replay the reply.

        Purpose: enforces the full provider contract per request:
            authorization gate, 256 KiB request/response caps, no key
            leakage into the payload, pre-request spending check,
            single-choice finish reasons (stop/tool_calls only),
            assistant schema plus tool-call ID/name/argument shapes, and
            duplicate-ID rejection. Length cutoffs raise
            CompletionLimitError so incomplete synthesis is never
            mistaken for an accepted answer.
        Args:
            messages: privacy-filtered conversation so far.
            tools: model-facing tool definitions.
        Returns:
            Assistant message dict replaying the documented fields.
        Raises:
            ModelError: on disabled live access, missing key, oversize
                payloads, HTTP/transport failure, or schema violations
                (sanitized: no keys or bodies retained).
            CompletionLimitError: when finish_reason is length.
        Flow: Receives privacy-filtered messages and tools from run_investigation(); returns assistant message dict for its next local decision."""
        if self._transport is None and not self._live_authorized:
            raise ModelError('Live Meta requests are disabled: explicit paid-call authorization is required')
        key = 'test-only' if self._transport is not None else os.environ.get('MODEL_API_KEY')
        if not key and self._transport is None:
            raise ModelError('MODEL_API_KEY is not configured')
        payload = {'model': MODEL, 'messages': messages, 'tools': tools, 'tool_choice': 'auto',
                   'parallel_tool_calls': False, 'max_completion_tokens': self.completion_tokens, 'stream': False, 'store': False}
        entry = None
        started = time.monotonic()
        try:
            if not self.automatic_runtime and len(json.dumps(payload, allow_nan=False).encode()) > MAX_BYTES:
                raise ContextLimitError('Meta request exceeded the bounded context size')
            with httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=5, write=15, pool=5), follow_redirects=False,
                              trust_env=False, transport=self._transport) as client:
                size = len(client.build_request('POST', ENDPOINT, json=payload).content)
                # The key must never appear inside the payload itself; that would mean prompt-injected exfiltration.
                if not self.automatic_runtime and size > MAX_BYTES:
                    raise ContextLimitError('Meta request exceeded the bounded context size')
                if self._transport is None and key and key in json.dumps(payload):
                    raise ModelError('Meta outbound boundary blocked the request')
                if self.max_estimated_cost_usd is not None and projected_token_cost(
                        self.telemetry, size, self.completion_tokens) > self.max_estimated_cost_usd:
                    self.budget_blocked = True
                    raise ModelError('Estimated spending ceiling would be exceeded; no request was sent')
                entry = RequestTelemetry(request_number=len(self.telemetry)+1, completion_allowance=self.completion_tokens,
                    http_status=None, finish_reason=None, prompt_tokens=None, cached_input_tokens=None,
                    completion_tokens=None, reasoning_tokens=None, visible_output_tokens=None, total_tokens=None,
                    elapsed_seconds=0.0, request_bytes=size, model_matches=None, error_code=None)
                self.telemetry.append(entry)
                with client.stream('POST', ENDPOINT, json=payload,
                                   headers={'Authorization': f'Bearer {key or "test-only"}'}) as response:
                    entry['http_status'] = response.status_code
                    if response.status_code != 200:
                        raise ModelError(f'Meta returned HTTP {response.status_code}; no response body retained')
                    if response.headers.get('content-type', '').split(';')[0] != 'application/json':
                        raise ModelError('Meta returned an unsupported content type')
                    body = bytearray()
                    for chunk in response.iter_bytes(chunk_size=8192):
                        response_limit = MAX_AUTOMATIC_RESPONSE_BYTES if self.automatic_runtime else MAX_BYTES
                        if len(body) + len(chunk) > response_limit:
                            raise ModelError('Meta response exceeded the bounded size')
                        body.extend(chunk)
                    data = json.loads(body)
                    if isinstance(data, dict): record_usage(entry, data, MODEL)
            choices = data['choices']
            if len(choices) == 1 and choices[0]['finish_reason'] == 'length':
                raise CompletionLimitError('Meta reached the configured completion-token limit; final synthesis is incomplete')
            if len(choices) != 1 or choices[0]['finish_reason'] not in ('stop', 'tool_calls'):
                raise ModelError('Meta returned an incomplete or unsupported completion')
            message = choices[0]['message']
            if message['role'] != 'assistant' or not (message.get('content') is None or isinstance(message['content'], str)):
                raise ModelError('Meta returned an unsupported assistant schema')
            for field in ('refusal', 'reasoning_content', 'reasoning'):
                if message.get(field) is not None and not isinstance(message[field], str):
                    raise ModelError('Meta returned an unsupported optional assistant field')
            calls = message.get('tool_calls') or []
            if not isinstance(calls, list) or (not self.automatic_runtime and len(calls) > 8):
                raise ModelError('Meta returned an invalid tool-call batch')
            for call in calls:
                if (call['type'] != 'function' or not isinstance(call['id'], str) or not CALL_ID.fullmatch(call['id']) or
                    not isinstance(call['function']['name'], str) or not isinstance(call['function']['arguments'], str)):
                    raise ModelError('Meta returned an invalid tool-call schema')
            if len({call['id'] for call in calls}) != len(calls):
                raise ModelError('Meta returned duplicate call identifiers')
            # Replay every documented assistant field, including tool calls and reasoning if present.
            fields = ('role', 'content', 'tool_calls', 'refusal', 'reasoning_content', 'reasoning', 'name')
            return {field: message[field] for field in fields if field in message}
        except ModelError as error:
            if entry is not None:
                entry['error_code'] = error.code if isinstance(error, CompletionLimitError) else 'PROVIDER_FAILURE'
            raise
        except httpx.TimeoutException:
            if entry is not None: entry['error_code'] = 'TIMEOUT'
            raise ModelError('Meta request timed out') from None
        except httpx.HTTPError:
            if entry is not None: entry['error_code'] = 'TRANSPORT_FAILURE'
            raise ModelError('Meta network or TLS request failed') from None
        except (KeyError, IndexError, TypeError, ValueError, RecursionError):
            if entry is not None: entry['error_code'] = 'INVALID_PROTOCOL'
            raise ModelError('Meta returned invalid JSON or an unsupported response schema') from None
        finally:
            if entry is not None: entry['elapsed_seconds'] = round(time.monotonic() - started, 6)
