"""HTTP contracts and real ASGI routing exercised with TestClient."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from api.app import app
from api.routes.query import QueryRequest


class StubPipeline:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.received = []

    async def answer(self, **kwargs):
        self.received.append(kwargs)
        if self.should_fail:
            raise RuntimeError("internal secret stacktrace")
        return {"answer": "По инструкции проверьте питание.", "sources": [],
                "status": "approved", "latency": 0.02, "figures": []}


@pytest.fixture
def client(monkeypatch):
    import api.routes.query as query
    stub = StubPipeline()
    monkeypatch.setattr(query, "get_pipeline", lambda: stub)
    with TestClient(app) as c:
        yield c, stub


def test_query_auth_fails_closed(client, monkeypatch):
    c, stub = client
    monkeypatch.delenv("KOIB_API_KEY", raising=False)
    monkeypatch.delenv("KOIB_ALLOW_UNAUTHENTICATED", raising=False)
    assert c.post("/query", json={"query": "Тест"}).status_code == 503
    monkeypatch.setenv("KOIB_API_KEY", "super-secure-key")
    assert c.post("/query", json={"query": "Тест"}).status_code == 401
    ok = c.post("/query", json={"query": "Тест", "validate": False,
                                 "model_filter": "koib2010", "top_k": 3},
                headers={"X-API-Key": "super-secure-key"})
    assert ok.status_code == 200
    assert ok.json()["status"] == "approved"
    assert stub.received[-1]["validate"] is False
    assert stub.received[-1]["model_filter"] == "koib2010"


def test_query_validates_input_and_strips_unexpected(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("KOIB_ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.delenv("KOIB_API_KEY", raising=False)
    for body in ({"query": ""}, {"query": "a", "top_k": 100},
                 {"query": "a", "model_filter": "evil"},
                 {"query": "a", "unknown": "unexpected"}):
        assert c.post("/query", json=body).status_code == 422


def test_query_does_not_expose_server_exception(client, monkeypatch):
    c, _ = client
    import api.routes.query as query
    monkeypatch.setenv("KOIB_ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.delenv("KOIB_API_KEY", raising=False)
    monkeypatch.setattr(query, "get_pipeline", lambda: StubPipeline(should_fail=True))
    resp = c.post("/query", json={"query": "test"})
    assert resp.status_code == 503
    assert "internal secret" not in resp.text


def test_liveness_and_readiness_missing(client, monkeypatch, tmp_path):
    c, _ = client
    import api.routes.health as health
    monkeypatch.setattr(health, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(health, "DOCSTORE_DIR", tmp_path / "docstore")
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/ready")
    assert r.status_code == 503 and r.json()["status"] == "not_ready"


def test_readiness_with_real_sqlite(client, monkeypatch, tmp_path):
    c, _ = client
    import api.routes.health as health
    from src.indexing import BM25FTSIndex, DocStore
    from tests.test_search_and_cache import chunk
    idx = tmp_path / "index"
    dsdir = tmp_path / "docstore"
    bm = BM25FTSIndex(idx / "bm25_fts.db")
    ds = DocStore(dsdir / "docstore.db")
    try:
        bm.add_chunks([chunk("a", "сканер считывает бюллетень")])
        ds.add_many([chunk("a", "сканер считывает бюллетень")])
        monkeypatch.setattr(health, "INDEX_DIR", idx)
        monkeypatch.setattr(health, "DOCSTORE_DIR", dsdir)
        r = c.get("/ready")
        assert r.status_code == 200 and r.json()["status"] == "degraded"
        (idx / "text_index.faiss").write_bytes(b"placeholder")
        assert c.get("/ready").json()["status"] == "degraded"
        monkeypatch.setenv("KOIB_TRUST_FAISS_PICKLE", "true")
        assert c.get("/ready").json()["status"] == "ready"
    finally:
        bm.close()
        ds.close()


def test_validation_alias():
    assert QueryRequest(query="test", validate=False).validate_answer is False
    assert QueryRequest(query="test").validate_answer is True


def test_full_http_to_fts_and_stub_llm(client, monkeypatch, tmp_path):
    """Actual API -> pipeline -> retriever -> SQLite FTS -> fake network LLM."""
    import api.routes.query as route
    import src.indexing as indexing
    import src.rag_pipeline as pipeline_mod
    from tests.test_search_and_cache import chunk
    from tests.test_pipeline import LLM, Cache, Memory
    from src.indexing import IndexBuilder
    from src.retrieval import HybridRetriever
    from src.rag_pipeline import RAGPipeline
    monkeypatch.setattr(indexing, "BM25_USE_LEMMATIZATION", False)
    monkeypatch.setattr(indexing, "get_global_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(pipeline_mod, "get_global_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(pipeline_mod, "ANSWER_REFINE_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "PROCEDURAL_REMINDER_ENABLED", False)
    monkeypatch.setenv("KOIB_API_KEY", "test-secure-key")
    builder = IndexBuilder(index_dir=tmp_path / "index")
    docs = [chunk("full", "Сканер считывает бумажный бюллетень", "koib2010")]
    builder.bm25.add_chunks(docs)
    builder.docstore.add_many(docs)
    retriever = HybridRetriever(builder)
    llm = LLM("Проверьте сканер. [Документ: manual.pdf, стр. 2]")
    pipeline = RAGPipeline(retriever=retriever, llm=llm, semantic_cache=Cache(), memory=Memory())
    monkeypatch.setattr(route, "get_pipeline", lambda: pipeline)
    c, _ = client
    response = c.post("/query", json={"query": "как проверить сканер бумажный бюллетень?", "validate": False,
                                      "model_filter": "koib2010"}, headers={"X-API-Key": "test-secure-key"})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "approved"
    assert response.json()["sources"][0]["document"] == "manual.pdf"
    assert "бумажный" in llm.calls[0]
    import asyncio
    asyncio.run(pipeline.close())
