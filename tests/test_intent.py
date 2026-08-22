"""意图识别确定性逻辑测试(不发起 LLM/网络调用)。"""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from core.intent_recognizer import IntentRecognizer, IntentCategory


def _rec():
    # base_url 非空 → 禁用 Embedding;构造器不发起网络调用,只测同步纯函数。
    return IntentRecognizer(api_key="test-key", base_url="https://example.invalid")


def test_pattern_advice_request():
    r = _rec()
    for msg in ["帮我推荐一只能翻倍的股票", "明天会涨吗", "帮我下单买入", "保本吗"]:
        out = r._pattern_recognize(msg)
        assert out["intent"] == IntentCategory.ADVICE_REQUEST, msg


def test_pattern_research_and_compliance():
    r = _rec()
    assert r._pattern_recognize("帮我找这家公司的研报")["intent"] == IntentCategory.RESEARCH_REPORT
    assert r._pattern_recognize("我的风险测评是R2能买吗")["intent"] == IntentCategory.SUITABILITY
    assert r._pattern_recognize("这只ETF的跟踪误差多少")["intent"] == IntentCategory.PRODUCT_INFO


def test_entities_finance():
    r = _rec()
    e = r._extract_entities("600519 的 PE 是多少,我风险等级R3,今年涨了20%")
    assert "600519" in e["ticker"]
    assert any("PE" in m.upper() for m in e["metric"])
    assert "R3" in e["risk_level"]
    assert any("20%" in p for p in e["percentage"])


def test_intent_group_mapping():
    r = _rec()
    assert r._intent_group(IntentCategory.VALUATION) == "research"
    assert r._intent_group(IntentCategory.MARKET_QUOTE) == "market"
    assert r._intent_group(IntentCategory.SUITABILITY) == "compliance"
    assert r._intent_group(IntentCategory.ADVICE_REQUEST) == "escalation"


# ── 英文输入(双语关键词)────────────────────────────────────────────────────

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
