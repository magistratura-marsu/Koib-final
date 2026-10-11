from src.retrieval import RetrievalResult
from src.validation import AnswerValidator


def test_short_answer_without_citation_requires_review():
    chunk = RetrievalResult(chunk_id="a", content="Текст", source="manual.pdf", page=2)
    result = AnswerValidator().validate("Проверьте питание.", [chunk], "как проверить?")
    assert result.status == "review"


def test_fabricated_source_flagged_not_substring_matched():
    chunk = RetrievalResult(chunk_id="a", content="Текст", source="manual.pdf", page=2)
    result = AnswerValidator().validate(
        "Сделайте то-то. [Документ: fake-manual.pdf, стр. 2]", [chunk], "что делать?")
    assert result.status == "review"
    assert any(c.name == "citations" and not c.passed for c in result.checks)


def test_matching_source_approved():
    chunk = RetrievalResult(chunk_id="a", content="Текст", source="docs/manual.pdf", page=2)
    result = AnswerValidator().validate(
        "Проверьте питание. [Документ: manual.pdf, стр. 2]", [chunk], "что делать?")
    assert result.status == "approved"


def test_vk_callback_requires_secret_group_confirmation(monkeypatch):
    import src.vk_bot as vk
    service = object()  # validate_callback doesn't touch instance state
    incoming = {"secret": "strong-secret", "group_id": "123", "type": "confirmation"}
    monkeypatch.setattr(vk, "VK_SECRET_KEY", "")
    monkeypatch.setattr(vk, "VK_GROUP_ID", "")
    monkeypatch.setattr(vk, "VK_CONFIRM_CODE", "")
    assert not vk.VKBotService.validate_callback(service, incoming)
    monkeypatch.setattr(vk, "VK_SECRET_KEY", "strong-secret")
    monkeypatch.setattr(vk, "VK_GROUP_ID", "123")
    monkeypatch.setattr(vk, "VK_CONFIRM_CODE", "trusted-confirmation")
    assert vk.VKBotService.validate_callback(service, incoming)
    assert not vk.VKBotService.validate_callback(service, {**incoming, "secret": "bad"})
    assert not vk.VKBotService.validate_callback(service, {**incoming, "group_id": "456"})


def test_new_number_without_evidence_requires_review():
    chunk = RetrievalResult(chunk_id="c", content="Maximum load is 18 bar.", source="manual.pdf", page=2)
    validator = AnswerValidator()
    verdict = validator.validate("Максимум 89 бар [Документ: manual.pdf, стр. 2]", [chunk], "Каков максимум?")
    assert verdict.status == "review"
    assert any(c.name == "numeric_grounding" and not c.passed for c in verdict.checks)


def test_literal_number_from_evidence_can_pass():
    chunk = RetrievalResult(chunk_id="c", content="Maximum load is 18 bar.", source="manual.pdf", page=2)
    verdict = AnswerValidator().validate("Максимум 18 бар [Документ: manual.pdf, стр. 2]", [chunk])
    assert verdict.status == "approved"
