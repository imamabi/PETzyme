"""Local NREL data only; no transformation of scientific values."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import httpx

DOI = "10.5281/zenodo.15417757"
RECORD_URL = "https://zenodo.org/records/15417757"
METADATA_URL = "https://zenodo.org/api/records/15417757"
ARCHIVE_URL = "https://zenodo.org/api/records/15417757/files/p740.tar.gz/content"
MAX_ARCHIVE_BYTES = 100_000_000
MAX_CSV_BYTES = 20_000_000


class NrelError(RuntimeError):
    """Source validation or archive integrity error."""


def verify_checksum(path: Path, published: str) -> str:
    """Verify a downloaded archive against its published MD5 checksum.

    Purpose: proves the local bytes are exactly what Zenodo published
        before anything is extracted. Streams in 1 MiB chunks so the
        100 MB archive never sits fully in memory twice.
    Args:
        path: local Path of the downloaded archive.
        published: checksum string in 'md5:<32 hex>' form.
    Returns:
        The verified checksum string (equal to `published`).
    Raises:
        NrelError: when the checksum format is wrong or the digest differs.
    Flow: Receives external archive Path and published checksum str; returns verified checksum str for prepare()."""
    if not re.fullmatch(r"md5:[0-9a-f]{32}", published):
        raise NrelError("Expected a published md5 checksum")
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = "md5:" + digest.hexdigest()
    if actual != published:
        raise NrelError("NREL archive checksum mismatch")
    return actual


def extract_labels(archive: Path, destination: Path) -> str:
    """Extract only label_data.csv from the archive, rejecting unsafe members.

    Purpose: obtains the evidence file while blocking tar attacks: absolute
        paths, '..', backslashes, symlinks, and non-file members are all
        rejected, and member paths are never used as filesystem destinations.
        Writes atomically via a '.csv.part' file so a crash cannot leave a
        half-written CSV behind.
    Args:
        archive: Path of the verified p740.tar.gz.
        destination: Path where label_data.csv must land.
    Returns:
        The archive member name string that was extracted.
    Raises:
        NrelError: on symlink destinations, unsafe members, ambiguous or
            missing label_data.csv, or size-limit violations.
    Flow: Receives archive Path and destination Path from prepare(); returns member name str after safely writing only label_data.csv."""
    if destination.is_symlink():
        raise NrelError("Refusing a symlink destination")
    with tarfile.open(archive, "r:gz") as bundle:
        selected = None
        for index, member in enumerate(bundle):
            if index >= 100_000:
                raise NrelError("Too many archive members")
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise NrelError("Unsafe archive member path")
            if not (member.isfile() or member.isdir()):
                raise NrelError("Links and special archive members are forbidden")
            if path.name == "label_data.csv":
                if selected is not None or not member.isfile():
                    raise NrelError("Ambiguous or invalid label_data.csv")
                if not 0 < member.size <= MAX_CSV_BYTES:
                    raise NrelError("CSV exceeds size limit or is empty")
                selected = member
        if selected is None:
            raise NrelError("label_data.csv not found")
        # Never use extract/extractall or use member paths as filesystem destinations.
        source = bundle.extractfile(selected)
        if source is None:
            raise NrelError("CSV could not be read")
        with source:
            data = source.read(MAX_CSV_BYTES + 1)
        if len(data) != selected.size:
            raise NrelError("CSV size mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create plus atomic replace: a second concurrent prepare fails instead of interleaving bytes.
        partial = destination.with_suffix(".csv.part")
        try:
            with partial.open("xb") as handle:
                handle.write(data)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        return selected.name


def parse_csv_text(content: str) -> tuple[list[str], list[list[str]]]:
    """Parse CSV text into headers plus original string rows.

    Purpose: strict structural parse shared by every CSV reader. Uses
        strict csv parsing so malformed quoting fails loudly instead of
        silently shifting columns.
    Args:
        content: full CSV text, already decoded as UTF-8.
    Returns:
        (columns, rows) tuple of original strings; no cell is interpreted.
    Raises:
        NrelError: on missing/duplicate headers or ragged row widths.
    Example:
        >>> parse_csv_text(',sequence\\nE1,ACDEF\\n')
        (['', 'sequence'], [['E1', 'ACDEF']])
    Flow: Receives CSV text str from load_csv() or load_nrel(); returns tuple[list[str], list[list[str]]] of original cells for those loaders."""
    reader = csv.reader(io.StringIO(content, newline=''), strict=True)
    columns = next(reader, None)
    if not columns or len(set(columns)) != len(columns):
        raise NrelError("Missing or duplicate CSV headers")
    rows: list[list[str]] = []
    for row in reader:
        if len(row) != len(columns):
            raise NrelError("CSV row width differs from header")
        rows.append(row)
    return columns, rows


def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a local CSV file as original strings (UTF-8, strict parse).

    Purpose: thin file wrapper over parse_csv_text() for one-off local
        validation and the prepare() pipeline.
    Flow: Receives external local CSV Path; returns tuple[list[str], list[list[str]]] of original cells for prepare() and local validation."""
    return parse_csv_text(path.read_bytes().decode('utf-8'))


def prepare(root: Path) -> dict:
    """Download, verify, and extract the NREL dataset without replacing an existing manifest.

    Purpose: one-time acquisition pipeline (scripts.prepare_nrel). Fetches
        Zenodo record metadata, checks DOI/URL/size, streams the archive
        with size limits, verifies its checksum and extracts label_data.csv
        safely. Existing source and reconciliation manifests stay unchanged.
    Args:
        root: repository root Path; data lands under root/'data/nrel'.
    Returns:
        Manifest dict (also written to NREL_SOURCE_MANIFEST.json) with DOI,
            checksum, license, attribution, and validation details.
    Raises:
        NrelError: on DOI/URL mismatch, size or checksum failure.
        httpx.HTTPError: on network/TLS failure (uncaught, fails loudly).
    Limitations:
        Performs live network I/O; normal tool paths reuse the downloaded
        files and never call this.
    Flow: Receives external repository root Path; returns source manifest dict after writing the verified NREL archive and selected CSV."""
    manifest_path = root / "NREL_SOURCE_MANIFEST.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing is not None and (not isinstance(existing, dict) or existing.get("dataset_doi") != DOI):
        raise NrelError("Existing source manifest does not match the expected NREL record")
    directory = root / "data" / "nrel"
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(root.resolve()):
        raise NrelError("NREL data directory must stay inside the project")
    archive = directory / "p740.tar.gz"
    target = directory / "label_data.csv"
    if archive.is_symlink() or target.is_symlink():
        raise NrelError("Refusing symlinked source files")
    expected_csv = (existing or {}).get("unit_reconciliation", {}).get("csv_sha256")
    if expected_csv and target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected_csv:
        raise NrelError("Existing CSV does not match the pinned reconciliation certificate")
    if existing and expected_csv and archive.is_file() and target.is_file():
        verify_checksum(archive, existing["archive_checksum"])
        load_csv(target)
        return existing
    with httpx.Client(timeout=httpx.Timeout(60, connect=15),
                      follow_redirects=False, trust_env=False) as client:
        response = client.get(METADATA_URL)
        response.raise_for_status()
        record = response.json()
        metadata = record["metadata"]
        if metadata["doi"] != DOI:
            raise NrelError("Unexpected dataset DOI")
        files = [f for f in record["files"] if f["key"] == "p740.tar.gz"]
        if len(files) != 1:
            raise NrelError("Expected exactly one p740.tar.gz")
        info = files[0]
        if existing and info["checksum"] != existing.get("archive_checksum"):
            raise NrelError("Published archive does not match the existing source manifest")
        if info["links"]["self"] != ARCHIVE_URL:
            raise NrelError("Archive URL differs from verified official URL")
        if not 0 < info["size"] <= MAX_ARCHIVE_BYTES:
            raise NrelError("Archive exceeds size limit")
        if archive.exists():
            verify_checksum(archive, info["checksum"])
        partial = archive.with_suffix(".gz.part")
        created_download = False
        try:
            with client.stream("GET", ARCHIVE_URL) as download:
                download.raise_for_status()
                size = 0
                with partial.open("xb") as handle:
                    created_download = True
                    for chunk in download.iter_bytes():
                        size += len(chunk)
                        if size > info["size"]:
                            raise NrelError("Archive larger than published size")
                        handle.write(chunk)
            if size != info["size"]:
                raise NrelError("Archive size mismatch")
            checksum = verify_checksum(partial, info["checksum"])
            partial.replace(archive)
        finally:
            if created_download:
                partial.unlink(missing_ok=True)
    downloaded = datetime.now(timezone.utc).isoformat()
    candidate = directory / "label_data.csv.verified"
    if target.is_symlink() or candidate.exists() or candidate.is_symlink():
        raise NrelError("Refusing an unsafe or pre-existing extraction target")
    try:
        member = extract_labels(archive, candidate)
        if expected_csv and hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_csv:
            raise NrelError("Downloaded CSV does not match the pinned reconciliation certificate")
        columns, rows = load_csv(candidate)
        if target.exists():
            if hashlib.sha256(candidate.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                raise NrelError("Existing CSV differs from the verified source; it was not replaced")
        else:
            candidate.replace(target)
    finally:
        candidate.unlink(missing_ok=True)
    manifest = {
        "dataset_doi": DOI, "record_url": RECORD_URL,
        "record_version": "v1", "version_source": "Zenodo record page (API metadata.version absent)",
        "download_url": ARCHIVE_URL, "downloaded_at_utc": downloaded,
        "archive_checksum": checksum, "archive_bytes": size,
        "extracted_file": str(target.relative_to(root)), "archive_member": member,
        "license": metadata["license"], "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": {"title": metadata["title"],
                        "creators": [item["name"] for item in metadata["creators"]]},
        "validation": {"checksum": "passed", "safe_extraction": "passed",
                       "csv_structure": "passed", "columns": columns, "rows": len(rows),
                       "values": "Original strings; no units inferred or activities converted"},
    }
    if existing is not None:
        return existing
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
