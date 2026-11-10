from unittest.mock import patch
from app.extract import PROMPT, extract_triples


def test_prompt_contains_engineering_examples():
    """Verify prompt includes few-shot engineering examples for failure causation and impacts."""
    assert "Duplicate debit is caused by callback timeout" in PROMPT
    assert "callback timeout affects gateway x" in PROMPT.lower()
    assert '"s": "duplicate debit"' in PROMPT
    assert '"s_type": "Failure"' in PROMPT
    assert '"o": "callback timeout"' in PROMPT
    assert '"o_type": "Failure"' in PROMPT
    assert '"r": "CAUSED_BY"' in PROMPT
    assert '"r": "AFFECTS"' in PROMPT


def test_prompt_contains_negative_assertion_suppression():
    """Verify prompt explicitly instructs suppressing negative assertions and unrelated statements."""
    assert "NEGATIVE ASSERTION SUPPRESSION" in PROMPT
    assert "unrelated to any recent release" in PROMPT
    assert "do NOT generate any triple" in PROMPT or "do not generate" in PROMPT.lower()


def test_prompt_contains_failure_typing_guidance():
    """Verify prompt guides models to classify failure mechanisms as Failure, not Release."""
    assert "callback timeout" in PROMPT
    assert "NEVER type a timeout or failure as a Release" in PROMPT or "Failure" in PROMPT


def test_extract_triples_filters_invalid_and_reserved_triples():
    """Verify extract_triples filters out invalid, reserved, and self-loop triples returned by LLM."""
    mock_llm_response = {
        "triples": [
            # Valid engineering triple
            {"s": "duplicate debit", "s_type": "Failure", "r": "CAUSED_BY", "o": "callback timeout", "o_type": "Failure"},
            # Invalid: reserved ontology word as entity name
            {"s": "Gateway Y", "s_type": "Gateway", "r": "CHANGED_IN", "o": "Release", "o_type": "Release"},
            # Invalid: self-loop across types
            {"s": "Gateway Y", "s_type": "Failure", "r": "AFFECTS", "o": "Gateway Y", "o_type": "Gateway"},
            # Invalid: wrong relation endpoint types
            {"s": "callback timeout", "s_type": "Failure", "r": "DEPLOYED_ON", "o": "Gateway X", "o_type": "Gateway"},
            # Valid engineering triple
            {"s": "callback timeout", "s_type": "Failure", "r": "AFFECTS", "o": "Gateway X", "o_type": "Gateway"},
        ]
    }
    with patch("app.llm.chat_json", return_value=mock_llm_response):
        triples = extract_triples("Sample engineering text", cid="none")
        assert len(triples) == 2
        assert triples[0]["s"] == "duplicate debit" and triples[0]["r"] == "CAUSED_BY"
        assert triples[1]["s"] == "callback timeout" and triples[1]["r"] == "AFFECTS"
