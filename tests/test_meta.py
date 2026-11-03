import json
import httpx
import pytest
from petzyme.meta import ENDPOINT, MODEL, MetaModelClient, ModelError


def response(message=None, finish='stop'):
    return {'choices': [{'message': message or {'role': 'assistant', 'content': '{}', 'refusal': None},
                         'finish_reason': finish, 'index': 0}]}


def test_official_request_and_assistant_replay(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'TEST_SECRET_NEVER_LOG')
    calls = []
    message = {'role': 'assistant', 'content': None, 'refusal': None, 'reasoning_content': 'test-only reasoning',
               'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'load_nrel', 'arguments': '{}'}}]}
    def handler(request):
        calls.append(request)
        assert str(request.url) == ENDPOINT and request.method == 'POST'
        assert request.headers['authorization'] == 'Bearer test-only'
        body = json.loads(request.content)
        assert body['model'] == MODEL and body['store'] is False
        assert body['tool_choice'] == 'auto' and body['parallel_tool_calls'] is False
        assert body['messages'] == [{'role': 'user', 'content': 'test'}]
        return httpx.Response(200, json=response(message, 'tool_calls'))
    client = MetaModelClient(transport=httpx.MockTransport(handler))
    assert client.complete([{'role': 'user', 'content': 'test'}], []) == message
    assert len(calls) == 1


def test_no_paid_calls_even_with_key(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'TEST_SECRET')
    with pytest.raises(ModelError, match='disabled'):
        MetaModelClient().complete([], [])
    monkeypatch.delenv('MODEL_API_KEY')
    with pytest.raises(ModelError, match='not configured'):
        MetaModelClient(live_authorized=True).complete([], [])


@pytest.mark.parametrize('status', [301, 401, 429, 500])
def test_http_errors_redacted(status):
    api = MetaModelClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, text='PRIVATE_BODY')))
    with pytest.raises(ModelError) as error:
        api.complete([], [])
    assert 'PRIVATE_BODY' not in str(error.value)


def test_timeout():
    def handler(request): raise httpx.ReadTimeout('PRIVATE', request=request)
    with pytest.raises(ModelError, match='timed out'):
        MetaModelClient(transport=httpx.MockTransport(handler)).complete([], [])


@pytest.mark.parametrize('data', [{}, {'choices': []}, response(finish='length'),
                                  response({'role': 'user', 'content': 'wrong'}),
                                  response({'role': 'assistant', 'content': [], 'tool_calls': []})])
def test_malformed_and_incomplete(data):
    client = MetaModelClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data)))
    with pytest.raises(ModelError): client.complete([], [])


def test_oversized_response_and_optional_schema():
    for value in ({'role': 'assistant', 'content': '{}', 'reasoning_content': {}},):
        client = MetaModelClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response(value))))
        with pytest.raises(ModelError): client.complete([], [])
    client = MetaModelClient(transport=httpx.MockTransport(lambda r: httpx.Response(200,
        content=b'x' * 262145, headers={'content-type': 'application/json'})))
    with pytest.raises(ModelError, match='bounded size'): client.complete([], [])


def test_automatic_runtime_does_not_apply_legacy_request_byte_ceiling():
    seen = []
    def handler(request):
        seen.append(len(request.content))
        return httpx.Response(200, json=response())
    client = MetaModelClient(transport=httpx.MockTransport(handler), automatic_runtime=True,
                             completion_tokens=50000)
    assert client.complete([{'role': 'user', 'content': 'sample text ' * 25000}], [])['role'] == 'assistant'
    assert seen[0] > 262144
