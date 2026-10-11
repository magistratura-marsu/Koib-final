from src.retrieval import RetrievalResult
from tools.quality_gate import score_cases


class MockRetriever:
    def search(self, query, k=5, **kwargs):
        if query == "temperature":
            return [RetrievalResult("1", "710 Kelvin", source="region-A/guide.md", page=3)]
        return []


def test_quality_metrics_truthful():
    cases = [{"query": "temperature", "source": "region-A/guide.md", "contains": "710"},
             {"query": "unavailable", "source": "region-A/guide.md", "contains": "710"}]
    report = score_cases(MockRetriever(), cases, k=2)
    assert report["recall_at_k"] == 0.5
    assert report["mrr_at_k"] == 0.5
    assert report["failures"] == 1


def test_wrong_source_is_not_counted_as_hit():
    report = score_cases(MockRetriever(), [{"query": "temperature", "source": "region-B/guide.md"}])
    assert report["recall_at_k"] == 0
