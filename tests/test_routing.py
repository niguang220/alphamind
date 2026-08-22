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
