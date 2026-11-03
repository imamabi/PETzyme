import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from zymecycle.nrel import NrelError, extract_labels, load_csv, verify_checksum


def make_tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.REGTYPE:
                info.size = len(content)
            else:
                info.linkname = "../../outside"
            archive.addfile(info, io.BytesIO(content) if kind == tarfile.REGTYPE else None)


def test_checksum_and_preserved_csv(tmp_path):
    csv = tmp_path / "labels.csv"
    original = b'id,activity,note\r\n001,1.00,"a,b"\r\n002,,NA\r\n'
    csv.write_bytes(original)
    published = "md5:" + hashlib.md5(original).hexdigest()
    assert verify_checksum(csv, published) == published
    assert load_csv(csv) == (["id", "activity", "note"],
                             [["001", "1.00", "a,b"], ["002", "", "NA"]])
    csv.write_bytes(original + b"changed")
    with pytest.raises(NrelError, match="mismatch"):
        verify_checksum(csv, published)


@pytest.mark.parametrize("checksum", ["sha256:abc", "md5:bad", ""])
def test_bad_checksum(tmp_path, checksum):
    with pytest.raises(NrelError):
        verify_checksum(tmp_path / "missing", checksum)


def test_only_csv_extracted(tmp_path):
    archive = tmp_path / "test.tar.gz"
    make_tar(archive, [("p740/label_data.csv", b"id,value\n1,2.00\n", tarfile.REGTYPE),
                       ("p740/structure.pdb", b"do not extract", tarfile.REGTYPE)])
    output = tmp_path / "extracted" / "label_data.csv"
    assert extract_labels(archive, output) == "p740/label_data.csv"
    assert output.read_bytes() == b"id,value\n1,2.00\n"
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("name,kind", [
    ("../label_data.csv", tarfile.REGTYPE),
    ("/label_data.csv", tarfile.REGTYPE),
    ("p/../../outside", tarfile.REGTYPE),
    ("p\\label_data.csv", tarfile.REGTYPE),
    ("p/link", tarfile.SYMTYPE), ("p/link", tarfile.LNKTYPE),
    ("p/device", tarfile.CHRTYPE),
])
def test_unsafe_members_rejected_before_writing(tmp_path, name, kind):
    archive = tmp_path / "test.tar.gz"
    make_tar(archive, [("label_data.csv", b"id\n1\n", tarfile.REGTYPE),
                       (name, b"malicious", kind)])
    output = tmp_path / "out" / "label_data.csv"
    with pytest.raises(NrelError):
        extract_labels(archive, output)
    assert not output.exists()


def test_duplicate_or_missing_labels(tmp_path):
    archive = tmp_path / "test.tar.gz"
    for names in [["other.csv"], ["a/label_data.csv", "b/label_data.csv"]]:
        make_tar(archive, [(name, b"x\n1\n", tarfile.REGTYPE) for name in names])
        with pytest.raises(NrelError):
            extract_labels(archive, tmp_path / "labels.csv")


def test_symlink_destination(tmp_path):
    archive = tmp_path / "test.tar.gz"
    make_tar(archive, [("label_data.csv", b"a\n1\n", tarfile.REGTYPE)])
    target = tmp_path / "existing"
    target.write_text("untouched")
    output = tmp_path / "labels.csv"
    output.symlink_to(target)
    with pytest.raises(NrelError):
        extract_labels(archive, output)
    assert target.read_text() == "untouched"


@pytest.mark.parametrize("content", ["", "a,a\n1,2\n", "a,b\n1\n"])
def test_invalid_csv(tmp_path, content):
    path = tmp_path / "test.csv"
    path.write_text(content)
    with pytest.raises(NrelError):
        load_csv(path)


def test_downloaded_nrel_against_manifest():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "NREL_SOURCE_MANIFEST.json"
    archive = root / "data/nrel/p740.tar.gz"
    if not manifest_path.exists() or not archive.exists():
        pytest.skip("Run python -m scripts.prepare_nrel for the optional local-data check")
    manifest = json.loads(manifest_path.read_text())
    assert verify_checksum(archive, manifest["archive_checksum"])
    columns, rows = load_csv(root / manifest["extracted_file"])
    assert columns == manifest["validation"]["columns"]
    assert len(rows) == manifest["validation"]["rows"]


def test_shared_parser_preserves_quoted_newlines():
    from zymecycle.nrel import parse_csv_text
    assert parse_csv_text('id,note\r\n001,"keep\r\nnewline"\r\n') == (
        ['id', 'note'], [['001', 'keep\r\nnewline']])


def test_prepare_preserves_existing_reconciliation_and_is_idempotent(tmp_path, monkeypatch):
    import httpx
    from zymecycle import nrel

    csv = b'id,value\nE1,1.00\n'
    source = tmp_path / 'source.tar.gz'
    make_tar(source, [('p740/label_data.csv', csv, tarfile.REGTYPE)])
    archive = source.read_bytes()
    checksum = 'md5:' + hashlib.md5(archive).hexdigest()
    manifest = {'dataset_doi': nrel.DOI, 'archive_checksum': checksum,
                'unit_reconciliation': {'csv_sha256': hashlib.sha256(csv).hexdigest()},
                'working_unit_interpretation': {'certificate': 'preserve'}}
    path = tmp_path / 'NREL_SOURCE_MANIFEST.json'
    path.write_text(json.dumps(manifest, indent=2))
    before = path.read_bytes()
    requests = []
    def respond(request):
        requests.append(str(request.url))
        if str(request.url) == nrel.METADATA_URL:
            return httpx.Response(200, json={'metadata': {'doi': nrel.DOI, 'license': {'id':'cc-by-4.0'},
                'title':'synthetic', 'creators':[]}, 'files':[{'key':'p740.tar.gz', 'checksum':checksum,
                'size':len(archive), 'links':{'self':nrel.ARCHIVE_URL}}]})
        return httpx.Response(200, content=archive)
    client = httpx.Client
    monkeypatch.setattr(nrel.httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(respond)))
    assert nrel.prepare(tmp_path) == manifest
    assert path.read_bytes() == before
    assert (tmp_path / 'data/nrel/label_data.csv').read_bytes() == csv
    assert len(requests) == 2
    assert nrel.prepare(tmp_path) == manifest and len(requests) == 2
    assert path.read_bytes() == before


def test_prepare_rejects_existing_csv_hash_mismatch_without_changing_manifest(tmp_path):
    from zymecycle import nrel

    directory = tmp_path / 'data/nrel'
    directory.mkdir(parents=True)
    archive = directory / 'p740.tar.gz'
    archive.write_bytes(b'archive')
    (directory / 'label_data.csv').write_text('id\nwrong\n')
    path = tmp_path / 'NREL_SOURCE_MANIFEST.json'
    path.write_text(json.dumps({'dataset_doi': nrel.DOI,
        'archive_checksum': 'md5:' + hashlib.md5(b'archive').hexdigest(),
        'unit_reconciliation': {'csv_sha256': hashlib.sha256(b'different').hexdigest()}}))
    before = path.read_bytes()
    with pytest.raises(nrel.NrelError, match='pinned reconciliation'):
        nrel.prepare(tmp_path)
    assert path.read_bytes() == before
    archive.unlink()
    with pytest.raises(nrel.NrelError, match='pinned reconciliation'):
        nrel.prepare(tmp_path)
    assert path.read_bytes() == before
