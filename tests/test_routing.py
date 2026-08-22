"""Agent 路由与护栏的确定性逻辑测试(不发起 LLM/网络调用)。"""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from agents.agent_orchestrator import AgentType, AgentOrchestrator, Request
from core.intent_recognizer import IntentCategory


def _orch():
    return AgentOrchestrator(api_key="test-key", base_url="https://example.invalid")


def test_agent_type_values():
    assert {t.value for t in AgentType} == {"market", "research", "compliance", "escalation"}


def test_domain_scores_research():
    o = _orch()
    req = Request(message="帮我看下这家公司的估值和研报", user_id="u", conv_id="c",
                  intent=IntentCategory.VALUATION, entities={"metric": ["PE"]})
    scores = o._domain_scores(req)
    assert scores[AgentType.RESEARCH] == max(scores.values())


def test_domain_scores_compliance():
    o = _orch()
    req = Request(message="我风险测评R2,能不能开通两融", user_id="u", conv_id="c",
                  intent=IntentCategory.SUITABILITY, entities={"risk_level": ["R2"]})
    scores = o._domain_scores(req)
    assert scores[AgentType.COMPLIANCE] == max(scores.values())


def test_intent_routing_map():
    assert AgentOrchestrator._INTENT_ROUTING[IntentCategory.VALUATION] == AgentType.RESEARCH
    assert AgentOrchestrator._INTENT_ROUTING[IntentCategory.SUITABILITY] == AgentType.COMPLIANCE
    assert AgentOrchestrator._INTENT_ROUTING[IntentCategory.HUMAN_HANDOFF] == AgentType.ESCALATION


# ── 投资建议护栏(T3)────────────────────────────────────────────────────────

def test_guardrail_detects_advice():
    o = _orch()
    for msg in ["帮我推荐一只能翻倍的股票", "现在该不该买茅台", "明天会涨吗", "帮我下单"]:
        req = Request(message=msg, user_id="u", conv_id="c", intent=IntentCategory.ADVICE_REQUEST)
        assert o._needs_guardrail(req) is True, msg


def test_guardrail_keyword_without_intent():
    # 即便调用方给的意图不是 advice_request,关键词命中也应触发护栏
    o = _orch()
    req = Request(message="你觉得现在该买哪只白酒股", user_id="u", conv_id="c",
                  intent=IntentCategory.COMPARISON)
    assert o._needs_guardrail(req) is True


def test_guardrail_not_triggered_for_info():
    o = _orch()
    req = Request(message="沪深300ETF的费率是多少", user_id="u", conv_id="c",
                  intent=IntentCategory.PRODUCT_INFO)
    assert o._needs_guardrail(req) is False


def test_guardrail_response_shape():
    o = _orch()
    req = Request(message="推荐一只翻倍股", user_id="u", conv_id="c", intent=IntentCategory.ADVICE_REQUEST)
    res = o._guardrail_response(req, 0.0)
    assert res.escalated is True
    assert res.agent_type == AgentType.ESCALATION
    assert "不构成投资建议" in res.response
    assert "guardrail" in res.routing_reason
