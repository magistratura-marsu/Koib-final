"""End-to-end RAG orchestration using deterministic providers, no network calls."""
import asyncio
from src.retrieval import RetrievalResult
from src.rag_pipeline import RAGPipeline


class LLM:
    def __init__(self, value):
        self.value = value
        self.calls = []
        self.closed = False
    async def generate_async(self, prompt, **kwargs):
        self.calls.append(prompt)
        return self.value
    async def close(self):
        self.closed = True

class Search:
    def __init__(self, items):
        self.items = items
        self.calls = []
        self.closed = False
    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.items
    def close(self):
        self.closed = True

class Cache:
    def __init__(self):
        self.stored = []
        self.closed = False
    def get(self, *args, **kwargs):
        return None
    def set(self, *args, **kwargs):
        self.stored.append((args, kwargs))
    def close(self):
        self.closed = True

class Memory:
    async def get_history(self, user_id): return []
    async def add_message(self, *args): pass


def run_answer(monkeypatch, llm_text, hits, **params):
    import src.rag_pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "get_global_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(pipeline_mod, "ANSWER_REFINE_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "PROCEDURAL_REMINDER_ENABLED", False)
    retriever, llm, cache = Search(hits), LLM(llm_text), Cache()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, semantic_cache=cache, memory=Memory())
    async def exercise():
        try:
            result = await pipeline.answer("Как проверить сканер?", use_memory=False, validate=False, **params)
        finally:
            await pipeline.close()
        return result
    return asyncio.run(exercise()), retriever, llm, cache


def test_full_pipeline_with_offline_embedding_fallback(monkeypatch):
    hits = [RetrievalResult(chunk_id="a", content="Проверить подключение.", source="manual.pdf", page=5, score=0.5)]
    result, retriever, llm, cache = run_answer(monkeypatch, "См. [Документ: manual.pdf, стр. 5]", hits)
    assert result["status"] == "approved"
    assert result["sources"][0]["document"] == "manual.pdf"
    assert "manual.pdf" in llm.calls[0]
    assert retriever.closed and llm.closed and cache.closed
    assert not cache.stored  # validation disabled means no cache contamination


def test_no_evidence_never_calls_llm(monkeypatch):
    result, _, llm, _ = run_answer(monkeypatch, "unnecessary", [])
    assert result["status"] == "review"
    assert result["sources"] == [] and llm.calls == []


def test_llm_failure_no_raw_exception_leak(monkeypatch):
    hits = [RetrievalResult(chunk_id="a", content="Текст", source="manual.pdf", page=5)]
    result, _, _, _ = run_answer(monkeypatch, "Ошибка GigaChat: client secret=12345", hits)
    assert result["status"] == "review"
    assert "secret" not in result["answer"]


def test_disabled_semantic_cache_does_not_load_embedding(monkeypatch):
    import src.rag_pipeline as module
    monkeypatch.setattr(module, "SEMANTIC_CACHE_ENABLED", False)
    called = []
    def count_call():
        called.append(True)
        raise RuntimeError("must never load")
    monkeypatch.setattr(module, "get_global_embeddings", count_call)
    monkeypatch.setattr(module, "PROCEDURAL_REMINDER_ENABLED", False)
    monkeypatch.setattr(module, "ANSWER_REFINE_ENABLED", False)
    pipe = RAGPipeline(retriever=Search([]), llm=LLM(""),
                       semantic_cache=Cache(), memory=Memory())
    async def execute():
        result = await pipe.answer("Проверка", use_memory=False)
        await pipe.close()
        return result
    assert asyncio.run(execute())["status"] == "review"
    assert called == []
