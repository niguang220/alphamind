"""Deterministic intent-recognition tests (no LLM / network calls)."""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from core.intent_recognizer import IntentRecognizer, IntentCategory


def _rec():
    # A non-empty base_url disables Embedding; the constructor makes no network call,
    # so these tests exercise the synchronous pure functions only.
    return IntentRecognizer(api_key="test-key", base_url="https://example.invalid")


def test_intent_group_mapping():
    r = _rec()
    assert r._intent_group(IntentCategory.VALUATION) == "research"
    assert r._intent_group(IntentCategory.MARKET_QUOTE) == "market"
    assert r._intent_group(IntentCategory.SUITABILITY) == "compliance"
    assert r._intent_group(IntentCategory.ADVICE_REQUEST) == "escalation"


def test_pattern_advice_request_english():
    r = _rec()
    for msg in ["recommend me a stock that will double", "which one should I buy now",
                "will it go up tomorrow", "is it guaranteed to make money"]:
        out = r._pattern_recognize(msg)
        assert out["intent"] == IntentCategory.ADVICE_REQUEST, msg


def test_pattern_english_research_compliance():
    r = _rec()
    assert r._pattern_recognize("find me the research report on this company")["intent"] == IntentCategory.RESEARCH_REPORT
    assert r._pattern_recognize("my risk assessment is R2, can I buy this")["intent"] == IntentCategory.SUITABILITY


def test_entities_english():
    r = _rec()
    e = r._extract_entities("600519 P/E is 30, my risk level R3, up 20% this year")
    assert "600519" in e["ticker"]
    assert any("P/E" in m.upper() for m in e["metric"])
    assert "R3" in e["risk_level"]
    assert any("20%" in p for p in e["percentage"])


def test_entities_reject_lookalikes():
    """Metrics and tickers must not fire on substrings or plain acronyms."""
    r = _rec()
    e = r._extract_entities("is the CSI 300 ETF's valuation expensive? can I open margin trading")
    # "PE" hides inside "expensive" and "open"; neither is a metric.
    assert e["metric"] == [], e["metric"]
    # CSI / ETF / ROE are acronyms, not exchange codes.
    assert e["ticker"] == [], e["ticker"]


def test_entities_exchange_suffixes():
    r = _rec()
    assert "600519.SH" in r._extract_entities("compare 600519.SH with peers")["ticker"]
    assert "00700.HK" in r._extract_entities("what about 00700.HK")["ticker"]
