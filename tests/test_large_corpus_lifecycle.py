"""Index update invariants and failure isolation for a growing corpus."""
from pathlib import Path
import json

import pytest

from batch_ingest import BatchIngester
from src.indexing import IndexBuilder
from src.retrieval import HybridRetriever
from src.validation import AnswerValidator


def test_add_replace_delete_documents_without_full_rebuild(tmp_path, monkeypatch):
    monkeypatch.setenv("KOIB_BUILD_VECTORS", "false")
    docs = tmp_path / "docs"
    output = tmp_path / "output"
    (docs / "region-A").mkdir(parents=True)
    (docs / "region-B").mkdir(parents=True)
    first = docs / "region-A" / "guide.md"
    second = docs / "region-B" / "guide.md"
    first.write_text("Reactor HX90 operating temperature has reference value 625 Kelvin. " * 5)
    second.write_text("Pressure vessel LZ27 maximum allowable load is 18 bar. " * 5)
    batch = BatchIngester(docs_dir=docs, output_dir=output)
    batch.process_all()
    batch.index_builder.close()

    store = IndexBuilder(output / "index", output / "docstore" / "docstore.db")
    retr = HybridRetriever(store)
    try:
        assert any(r.source == "region-A/guide.md" for r in retr.search("HX90 625 Kelvin"))
        assert any(r.source == "region-B/guide.md" for r in retr.search("LZ27 18 bar"))
    finally:
        retr.close()

    first.write_text("Reactor HX90 operating temperature has reference value 710 Kelvin. " * 5)
    batch = BatchIngester(docs_dir=docs, output_dir=output)
    batch.process_all()
    batch.index_builder.close()
    store = IndexBuilder(output / "index", output / "docstore" / "docstore.db")
    retr = HybridRetriever(store)
    try:
        assert any("710" in (r.content or "") for r in retr.search("HX90 710 Kelvin"))
        assert not any("625" in (r.content or "") for r in retr.search("HX90 625 Kelvin"))
        assert any(r.source == "region-B/guide.md" for r in retr.search("LZ27 18 bar"))
    finally:
        retr.close()

    second.unlink()
    batch = BatchIngester(docs_dir=docs, output_dir=output)
    batch.process_all()
    batch.index_builder.close()
    store = IndexBuilder(output / "index", output / "docstore" / "docstore.db")
    retr = HybridRetriever(store)
    try:
        assert not any(r.source == "region-B/guide.md" for r in retr.search("LZ27 18 bar"))
    finally:
        retr.close()
    manifest = json.loads((output / "metadata" / "ingest_manifest.json").read_text())
    assert "doc:region-B/guide.md" not in manifest


def test_vector_index_protection_when_untrusted(tmp_path, monkeypatch):
    monkeypatch.setenv("KOIB_BUILD_VECTORS", "true")
    monkeypatch.delenv("KOIB_TRUST_FAISS_PICKLE", raising=False)
    p = tmp_path / "i"
    p.mkdir()
    (p / "text_index.faiss").write_bytes(b"fake")
    builder = IndexBuilder(p)
    from src.chunking import Chunk
    try:
        with pytest.raises(RuntimeError, match="could not be loaded safely"):
            builder.add_chunks([Chunk("a", "Some searchable text", metadata={"source": "demo.md"})])
    finally:
        builder.close()


def test_validator_does_not_approve_invented_citation():
    from src.retrieval import RetrievalResult
    real = [RetrievalResult(chunk_id="a", content="Pressure is 18 bar", source="manual.pdf", page=6)]
    verdict = AnswerValidator().validate("Pressure 18 bar [Документ: false.pdf, стр. 999]", real)
    assert verdict.status == "review"


def test_deleting_every_document_clears_lexical_index(tmp_path, monkeypatch):
    monkeypatch.setenv("KOIB_BUILD_VECTORS", "false")
    docs = tmp_path / "docs"
    output = tmp_path / "output"
    docs.mkdir()
    doc = docs / "only.md"
    doc.write_text("Reliable identifier ZX9810 and measured result 41", encoding="utf-8")
    builder = BatchIngester(docs_dir=docs, output_dir=output)
    builder.process_all()
    builder.index_builder.close()
    doc.unlink()
    builder = BatchIngester(docs_dir=docs, output_dir=output)
    builder.process_all()
    builder.index_builder.close()
    check = IndexBuilder(output / "index", output / "docstore" / "docstore.db")
    try:
        assert check.docstore.count() == 0
        assert check.bm25.count() == 0
        manifest = json.loads((output / "metadata" / "ingest_manifest.json").read_text())
        assert manifest == {}
    finally:
        check.close()
