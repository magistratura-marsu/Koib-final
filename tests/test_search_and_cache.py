"""Real SQLite FTS integration tests without heavyweight ML downloads."""
import concurrent.futures
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.indexing import BM25FTSIndex, DocStore, IndexBuilder, prepare_fts_query
from src.retrieval import HybridRetriever, SemanticCache


def chunk(chunk_id, text, model="koib2010", source="manual.pdf", seq=1):
    meta = {"source": source, "page": 2, "heading": "Проверка", "model": model, "seq": seq}
    return SimpleNamespace(chunk_id=chunk_id, content=text, full_content=None,
                           chunk_type="text", metadata=meta)


def test_fts5_index_and_docstore_real_sqlite(tmp_path, monkeypatch):
    import src.indexing as indexing
    monkeypatch.setattr(indexing, "BM25_USE_LEMMATIZATION", False)
    bm25 = BM25FTSIndex(tmp_path / "index" / "bm25_fts.db")
    ds = DocStore(tmp_path / "docstore" / "docstore.db")
    docs = [chunk("a", "Сканер аппарата проверяет бумажный бюллетень"),
            chunk("b", "Кабель питания проверяется при включении", seq=2)]
    try:
        bm25.add_chunks(docs)
        ds.add_many(docs)
        results = bm25.search("сканер бюллетень", k=3)
        assert results and results[0][0]["chunk_id"] == "a"
        assert "Сканер" in results[0][0]["content"]
        assert ds.get_content("a").startswith("Сканер")
        assert len(ds.get_neighbors("manual.pdf", 1, 1)) == 1
        bm25.add_chunks([chunk("a", "Сканер проверен")])
        assert bm25.count() == 2  # upsert, no duplicates
    finally:
        bm25.close()
        ds.close()


def test_query_builder_escapes_syntax_and_no_empty_query():
    assert prepare_fts_query("!!!") == ""
    assert 'OR' in prepare_fts_query("кабель экран")
    assert '"' in prepare_fts_query('"сканер" OR *')


def test_bm25_only_retriever_does_not_load_models(tmp_path, monkeypatch):
    import src.indexing as indexing
    monkeypatch.setattr(indexing, "BM25_USE_LEMMATIZATION", False)
    monkeypatch.setattr(indexing, "get_global_embeddings", lambda: (_ for _ in ()).throw(AssertionError("model load")))
    builder = IndexBuilder(index_dir=tmp_path / "index")
    c = chunk("a", "Сканер принимает бюллетени", "koib2010")
    builder.bm25.add_chunks([c])
    builder.docstore.add_many([c])
    r = HybridRetriever(builder)
    try:
        hits = r.search("сканер бюллетени", k=2, model_filter="koib2010")
        assert hits and hits[0].source == "manual.pdf"
        assert hits[0].score > 0
        assert r.search("сканер бюллетени", model_filter="koib2017a") == []
    finally:
        r.close()


def test_index_does_not_unpickle_without_explicit_trust(tmp_path, monkeypatch):
    import src.indexing as indexing
    monkeypatch.setattr(indexing, "get_global_embeddings", lambda: (_ for _ in ()).throw(AssertionError("model load")))
    builder = IndexBuilder(index_dir=tmp_path / "index")
    (builder.output_dir / "text_index.faiss").write_bytes(b"not actually faiss")
    monkeypatch.delenv("KOIB_TRUST_FAISS_PICKLE", raising=False)
    try:
        builder.load()  # should not load any model or deserialize pickle
        assert builder.text_vectorstore is None
    finally:
        builder.close()


def test_semantic_cache_dimension_guard_and_closes(tmp_path, monkeypatch):
    import src.retrieval as retrieval
    monkeypatch.setattr(retrieval, "SEMANTIC_CACHE_ENABLED", True)
    cache = SemanticCache(path=tmp_path / "cache.db", threshold=0.99)
    try:
        cache.set("кабель", [1.0, 0.0], "Ответ", [{"document": "a"}])
        assert cache.get("кабель", [1.0, 0.0])["answer"] == "Ответ"
        assert cache.get("кабель", [1.0, 0.0], model_filter="koib2010") is None
        assert cache.get("кабель", [1.0, 0.0], k=6) is None
        assert cache.get("кабель", [1.0, 0.0, 0.0]) is None
    finally:
        cache.close()


def test_parallel_bm25_reads(tmp_path, monkeypatch):
    import src.indexing as indexing
    monkeypatch.setattr(indexing, "BM25_USE_LEMMATIZATION", False)
    bm25 = BM25FTSIndex(tmp_path / "fts.db")
    bm25.add_chunks([chunk("a", "Сканер считывает бюллетень")])
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(lambda _: bm25.search("сканер"), range(40)))
        assert all(out and out[0][0]["chunk_id"] == "a" for out in outputs)
    finally:
        bm25.close()


def test_semantic_cache_invalidated_when_corpus_changes(tmp_path, monkeypatch):
    import src.retrieval as retrieval
    monkeypatch.setattr(retrieval, "SEMANTIC_CACHE_ENABLED", True)
    monkeypatch.setattr(retrieval, "OUTPUT_DIR", tmp_path / "output")
    index_file = tmp_path / "output" / "index_manifest.json"
    index_file.parent.mkdir()
    index_file.write_text("version one")
    cache = retrieval.SemanticCache(path=tmp_path / "cache.db")
    try:
        cache.set("Question", [0.0, 1.0], "Old answer", [])
        assert cache.get("Question", [0.0, 1.0]) is not None
        index_file.write_text("changed document index")
        assert cache.get("Question", [0.0, 1.0]) is None
    finally:
        cache.close()
