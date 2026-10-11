"""Archive validation and confinement invariants."""
import json
import zipfile
from pathlib import Path

import pytest

from tools.index_transfer import _validate_archive_entries, pack, unpack


def test_zip_slip_and_adjacent_directory_rejected(tmp_path):
    for evil in ("../escape.txt", "output/../escape.txt", "output2/escape.txt", "/tmp/escape.txt"):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(evil, "oops")
        with zipfile.ZipFile(archive) as z:
            with pytest.raises(ValueError):
                _validate_archive_entries(z)


def test_unpack_rejects_corrupt_manifest_without_writing(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("output/index_manifest.json", json.dumps({"files": {"index/a.db": {"size": 1, "sha256": "00"}}}))
        z.writestr("output/index/a.db", "x")
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="Контрольная сумма"):
        unpack(archive, out)
    assert not out.exists()


def test_figure_paths_confined(tmp_path, monkeypatch):
    import config
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "real.png").write_bytes(b"image")
    (tmp_path / "private.txt").write_text("TOP SECRET")
    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    assert config.resolve_figure_path("figures/real.png") == str(figures / "real.png")
    assert config.resolve_figure_path("private.txt") == ""
    assert config.resolve_figure_path("../../private.txt") == ""
    assert config.resolve_figure_path(str(tmp_path / "private.txt")) == ""
    (figures / "unsafe.png").symlink_to(tmp_path / "private.txt")
    assert config.resolve_figure_path("figures/unsafe.png") == ""


def test_distributions_maintain_deploy_layout(tmp_path):
    from make_apps import build_server
    dest = tmp_path / "server"
    build_server(dest)
    assert (dest / "deploy" / "Dockerfile").is_file()
    assert (dest / "deploy" / "docker-compose.yml").is_file()
    assert (dest / "requirements-server.txt").is_file()
    assert (dest / ".dockerignore").is_file()
    assert not any(dest.rglob("__pycache__"))


def test_pack_unpack_roundtrip_real_fts_and_checksums(tmp_path, monkeypatch):
    from src.indexing import BM25FTSIndex, DocStore
    from tests.test_search_and_cache import chunk
    import src.indexing as indexing
    monkeypatch.setattr(indexing, "BM25_USE_LEMMATIZATION", False)
    source = tmp_path / "builder" / "output"
    target = tmp_path / "deployed" / "output"
    bm = BM25FTSIndex(source / "index" / "bm25_fts.db")
    docstore = DocStore(source / "docstore" / "docstore.db")
    docs = [chunk("manual_1", "Сканер считывает бюллетень")]
    bm.add_chunks(docs)
    docstore.add_many(docs)
    bm.close()
    docstore.close()
    archive = tmp_path / "real-index.zip"
    pack(source, archive)
    unpack(archive, target)
    assert target.joinpath("index_manifest.json").is_file()
    assert BM25FTSIndex(target / "index" / "bm25_fts.db").search("сканер")
    # Packing again and corrupting a source file must abort before replacing target.
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(archive) as src_zip, zipfile.ZipFile(damaged, "w") as dst_zip:
        for name in src_zip.namelist():
            blob = src_zip.read(name)
            if name == "output/docstore/docstore.db":
                blob = b"corrupt"
            dst_zip.writestr(name, blob)
    with pytest.raises(ValueError, match="Контрольная сумма"):
        unpack(damaged, target, force=True)
    assert target.joinpath("docstore/docstore.db").is_file()
