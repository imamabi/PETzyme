import ast
import builtins
import inspect
import logging
import os

import httpx
import pytest
import petzyme.pazy as pazy
from petzyme.pazy import PazyClient, PazyError


def page(results, next_url=None):
    return {"count": 7, "next": next_url, "previous": None, "results": results}


def client(handler, monkeypatch):
    monkeypatch.setattr(pazy.time, "sleep", lambda delay: None)
    return PazyClient(transport=httpx.MockTransport(handler))


def test_request_construction(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path == "/api/proteins/7/":
            return httpx.Response(200, json={"id": 7, "synonyms": []})
        return httpx.Response(200, json=page([{"id": 7}]))
    api = client(handler, monkeypatch)
    assert api.search_pet_proteins(limit=1) == [{"id": 7}]
    assert api.get_protein(7)["id"] == 7
    assert api.get_literature(limit=1) == [{"id": 7}]
    assert [r.url.path for r in requests] == ["/api/proteins/", "/api/proteins/7/", "/api/literature/"]
    assert dict(requests[0].url.params) == {"substrate": "PET", "limit": "1", "offset": "0"}
    assert dict(requests[2].url.params) == {"limit": "1", "offset": "0"}
    assert all(r.method == "GET" and r.url.scheme == "https" and
               r.url.host == "api.pazy.eu" for r in requests)
    assert requests[0].extensions["timeout"] == {"connect": 5, "read": 15, "write": 15, "pool": 15}


def test_bounded_pagination_and_untrusted_next(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        n = int(request.url.params["limit"])
        return httpx.Response(200, json=page([{"id": i} for i in range(n)],
                                            "https://untrusted.invalid/bulk"))
    api = client(handler, monkeypatch)
    assert len(api.search_pet_proteins(limit=10, max_pages=2)) == 6
    assert [r.url.params["offset"] for r in requests] == ["0", "3"]
    requests.clear()
    assert len(api.search_pet_proteins(limit=4, max_pages=5)) == 4
    assert len(requests) == 2
    assert all(r.url.host == "api.pazy.eu" for r in requests)


@pytest.mark.parametrize("status", [301, 302, 400, 404, 429, 500, 503])
def test_http_errors_no_body_or_redirect(monkeypatch, status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="SYNTHETIC_SECRET_RESPONSE",
                              headers={"location": "http://untrusted.invalid"})
    api = client(handler, monkeypatch)
    with pytest.raises(PazyError, match=f"HTTP {status}") as error:
        api.search_pet_proteins()
    assert "SYNTHETIC_SECRET" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError])
def test_transport_errors(monkeypatch, error_type):
    def handler(request):
        raise error_type("SYNTHETIC_RESPONSE_MUST_NOT_LEAK", request=request)
    api = client(handler, monkeypatch)
    with pytest.raises(PazyError) as error:
        api.get_literature()
    assert "SYNTHETIC_RESPONSE" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("payload", [[], {}, {"results": []}, page("wrong"), page(["wrong"]),
                                      page([{}, {}, {}, {}])])
def test_schema_errors(monkeypatch, payload):
    api = client(lambda r: httpx.Response(200, json=payload), monkeypatch)
    with pytest.raises(PazyError):
        api.search_pet_proteins()


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"invalid", headers={"content-type": "application/json"}),
    httpx.Response(200, text="html"),
    httpx.Response(200, content=b"x" * (pazy.MAX_RESPONSE_BYTES + 1),
                   headers={"content-type": "application/json"}),
])
def test_bad_json_content_and_oversized_response(monkeypatch, response):
    api = client(lambda r: response, monkeypatch)
    with pytest.raises(PazyError):
        api.get_literature()


@pytest.mark.parametrize("path", ["/proteins/fasta/", "/proteins/csv/", "/proteins/json/",
                                   "/proteins/csv-sequences/", "/assays/", "/panda/",
                                   "https://panda.invalid/", "http://api.pazy.eu/api/proteins/",
                                   "/proteins/../literature/", "/proteins/1/?redirect=bad"])
def test_forbidden_routes_never_requested(monkeypatch, path):
    def forbidden(request):
        pytest.fail("A forbidden route reached HTTP transport")
    api = client(forbidden, monkeypatch)
    with pytest.raises(PazyError, match="allowlist"):
        api._get(path)


@pytest.mark.parametrize("value", [0, -1, 11, True, "1"])
def test_limit_bounds(monkeypatch, value):
    api = client(lambda r: pytest.fail("Invalid input reached transport"), monkeypatch)
    with pytest.raises(ValueError):
        api.search_pet_proteins(limit=value)


@pytest.mark.parametrize("value", [0, -1, 6, True, "1"])
def test_page_bounds(monkeypatch, value):
    api = client(lambda r: pytest.fail("Invalid input reached transport"), monkeypatch)
    with pytest.raises(ValueError):
        api.get_literature(max_pages=value)


@pytest.mark.parametrize("value", [0, -1, True, "1/../../bulk"])
def test_id_bounds(monkeypatch, value):
    api = client(lambda r: pytest.fail("Invalid input reached transport"), monkeypatch)
    with pytest.raises(ValueError):
        api.get_protein(value)


def test_detail_id_mismatch(monkeypatch):
    api = client(lambda r: httpx.Response(200, json={"id": 9}), monkeypatch)
    with pytest.raises(PazyError):
        api.get_protein(1)


def test_pacing(monkeypatch):
    waits = []
    monkeypatch.setattr(pazy.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(pazy.time, "sleep", waits.append)
    api = PazyClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=page([]))))
    api.get_literature()
    api.get_literature()
    assert waits == [0.5]


def test_no_persistence_cache_cookies_or_response_logs(monkeypatch, caplog, capsys, tmp_path):
    sentinel = "SYNTHETIC_SEQUENCE_NEVER_STORE"
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=page([{"id": 1, "amino_acid_sequence": sentinel}]),
                              headers={"set-cookie": "test=must-not-carry"})
    api = client(handler, monkeypatch)
    monkeypatch.chdir(tmp_path)
    def no_file(*args, **kwargs):
        pytest.fail("PAZy client attempted filesystem access")
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", no_file)
        guard.setattr(os, "open", no_file)
        with caplog.at_level(logging.DEBUG):
            first = api.search_pet_proteins(limit=1)
            second = api.search_pet_proteins(limit=1)
    assert first == second and first is not second
    assert len(requests) == 2
    assert all("cookie" not in request.headers for request in requests)
    assert list(tmp_path.iterdir()) == []
    assert sentinel not in caplog.text
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err
    assert set(vars(api)) == {"_transport", "_last_request"}
    tree = ast.parse(inspect.getsource(pazy))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    assert not imported & {"sqlite3", "pickle", "shelve", "logging", "pathlib"}


def test_documented_name_filter_is_only_candidate_selection(monkeypatch):
    def handler(request):
        assert request.url.path == '/api/proteins/'
        assert dict(request.url.params) == {'substrate': 'PET', 'name': 'LCC-ICCG', 'limit': '1', 'offset': '0'}
        return httpx.Response(200, json=page([]))
    api = client(handler, monkeypatch)
    assert api.search_pet_proteins(limit=1, name='LCC-ICCG') == []
    for invalid in ('', ' ' * 2, 'x' * 101, 123):
        with pytest.raises(ValueError):
            api.search_pet_proteins(name=invalid)
