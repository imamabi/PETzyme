"""PAZy live-only REST access. No file I/O, response logging, or record cache.

Source: PAZy — The Plastics-Active Enzymes Database, https://www.pazy.eu/
API documentation: https://www.pazy.eu/api-docs
Returned records are caller-owned, transient in-memory values. Never persist them.
"""
from __future__ import annotations

import re
import time

import httpx

BASE_URL = "https://api.pazy.eu/api"
SOURCE = "PAZy — The Plastics-Active Enzymes Database (https://www.pazy.eu/)"
MAX_RESULTS = 10
MAX_PAGES = 5
MAX_RESPONSE_BYTES = 262_144


class PazyError(RuntimeError):
    """Sanitized error; never carries a response, sequence, or raw exception."""


class PazyClient:
    """Read-only, bounded PAZy API client; live-only, nothing is persisted.

    Purpose: the single choke point for all PAZy protein/live access. Every
        request uses a fresh httpx client (no cookies or cross-request
        state), verified TLS, no redirects, 5s connect / 15s operation
        timeouts, a 256 KiB response cap, and 0.5s pacing between calls.
        Only the last-request timestamp is retained on the instance.
    Args:
        transport: optional httpx transport; tests pass a MockTransport so
            no network is touched. Production passes None (live HTTPS).
    Limitations:
        Returned dicts are caller-owned transient memory. Never write them
            to files, logs, caches, or transcripts.
    """

    source = SOURCE

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        """Store the transport override and zero the pacing clock.

        Flow: Receives external httpx.BaseTransport | None; returns None after initializing request controls for search_pet_proteins(), get_protein() and get_literature()."""
        self._transport = transport
        self._last_request = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Perform one paced, allowlisted GET and return its JSON object.

        Purpose: enforces every transport guard in one place: path
            allowlist, 0.5s pacing, fresh client, JSON content-type check,
            256 KiB streaming cap. All failures collapse to a sanitized
            PazyError that carries no response body or sequence.
        Raises:
            PazyError: on disallowed paths, HTTP errors, timeouts,
                oversize bodies, or non-object JSON.
        Flow: Receives allowed path str and parameters dict | None from _list() or get_protein(); returns transient response dict for those functions."""
        if not (path in ("/proteins/", "/literature/") or
                re.fullmatch(r"/proteins/[1-9][0-9]*/", path)):
            raise PazyError("Endpoint is outside the PAZy read-only allowlist")
        delay = 0.5 - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        self._last_request = time.monotonic()
        try:
            with httpx.Client(timeout=httpx.Timeout(15, connect=5),
                              follow_redirects=False, trust_env=False,
                              transport=self._transport,
                              headers={"Accept": "application/json"}) as client:
                with client.stream("GET", BASE_URL + path, params=params) as response:
                    if response.status_code != 200:
                        raise PazyError(f"PAZy returned HTTP {response.status_code}; try again later")
                    if response.headers.get("content-type", "").split(";")[0] != "application/json":
                        raise PazyError("PAZy returned an unexpected content type")
                    body = bytearray()
                    for chunk in response.iter_bytes(chunk_size=8192):
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise PazyError("PAZy response exceeded the small-response limit")
                        body.extend(chunk)
                    # json.loads here avoids storing the body on a client or a response cache.
                    import json
                    payload = json.loads(body)
        except httpx.TimeoutException:
            raise PazyError("PAZy request timed out; try again later") from None
        except httpx.HTTPError:
            raise PazyError("PAZy is unavailable (network or TLS error); try again later") from None
        except (ValueError, UnicodeError):
            raise PazyError("PAZy returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise PazyError("PAZy response schema changed: expected an object")
        return payload

    def _list(self, path: str, *, limit: int, max_pages: int, filters: dict) -> list[dict]:
        """Walk at most `max_pages` of a paginated list using tiny fixed pages.

        Purpose: bounds total retrieval (page size 3, hard total cap) and
            validates the pagination envelope on every page, ignoring any
            server-provided next-URL by computing offset locally.
        Raises:
            ValueError: when limit/max_pages are outside 1..10 / 1..5.
            PazyError: when the pagination schema or batch size is wrong.
        Flow: Receives path str, limits int and filters dict from search_pet_proteins() or get_literature(); returns list[dict] for those functions."""
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")
        if type(max_pages) is not int or not 1 <= max_pages <= MAX_PAGES:
            raise ValueError(f"max_pages must be between 1 and {MAX_PAGES}")
        records: list[dict] = []
        offset = 0
        for _ in range(max_pages):
            remaining = limit - len(records)
            # Tiny pages and a hard total cap; never follow server-provided URLs.
            page_size = min(3, remaining)
            payload = self._get(path, {**filters, "limit": page_size, "offset": offset})
            if not {"count", "next", "previous", "results"} <= payload.keys():
                raise PazyError("PAZy pagination schema changed")
            batch = payload["results"]
            if (not isinstance(batch, list) or any(not isinstance(r, dict) for r in batch)
                    or len(batch) > page_size or type(payload["count"]) is not int
                    or payload["count"] < 0
                    or not (payload["next"] is None or isinstance(payload["next"], str))):
                raise PazyError("PAZy returned an unexpected pagination structure or size")
            records.extend(batch)
            offset += len(batch)
            if len(records) >= limit or not batch or payload["next"] is None:
                break
        return records

    def search_pet_proteins(self, *, limit: int = 3, max_pages: int = 1, name: str | None = None) -> list[dict]:
        """List PET-substrate proteins, optionally filtered by exact name.

        Purpose: raw search behind the search_pazy_pet tool. Always pins
            substrate=PET; the optional name filter is a documented API
            filter, never an identity decision.
        Args:
            limit: 1..10 records wanted (tool callers use at most 3).
            max_pages: 1..5 pages to walk (tool callers use 1).
            name: optional nonempty candidate name, at most 100 chars.
        Returns:
            Transient list of raw protein dicts; empty when nothing matches.
        Raises:
            ValueError: on an invalid name argument.
            PazyError: on transport or schema failure (sanitized).
        Flow: Receives limit int, max_pages int and optional name str from search_pazy_pet() or external callers; returns transient list[dict] for search_pazy_pet() or smoke output."""
        filters = {"substrate": "PET"}
        if name is not None:
            if not isinstance(name, str) or not name.strip() or len(name) > 100:
                raise ValueError("name must be a nonempty string of at most 100 characters")
            filters['name'] = name
        return self._list("/proteins/", limit=limit, max_pages=max_pages, filters=filters)

    def get_protein(self, protein_id: int) -> dict:
        """Fetch one protein record and check the ID echoes back.

        Purpose: detail lookup behind get_pazy_metadata. Verifies the
            returned id equals the requested id so a server mix-up cannot
            silently attach the wrong sequence to a comparison.
        Args:
            protein_id: positive integer PAZy protein ID.
        Returns:
            Transient raw protein dict.
        Raises:
            ValueError: on a non-positive ID.
            PazyError: on transport failure or ID/schema mismatch.
        Flow: Receives selected protein_id int from get_pazy_metadata() or external callers; returns transient protein dict for get_pazy_metadata() or smoke validation."""
        if type(protein_id) is not int or protein_id < 1:
            raise ValueError("protein_id must be a positive integer")
        record = self._get(f"/proteins/{protein_id}/")
        if type(record.get("id")) is not int or record["id"] != protein_id:
            raise PazyError("PAZy detail schema changed or ID mismatched")
        return record

    def get_literature(self, *, limit: int = 3, max_pages: int = 1) -> list[dict]:
        """List general literature catalog entries (unassociated with proteins).

        Purpose: exposed only for smoke checks; the evidence path uses
            protein-embedded literature via get_associated_literature
            instead, so papers are never assigned by catalog guessing.
        Flow: Receives external bounded limit int and max_pages int; returns transient list[dict] as final unassociated catalog entries."""
        return self._list("/literature/", limit=limit, max_pages=max_pages, filters={})
