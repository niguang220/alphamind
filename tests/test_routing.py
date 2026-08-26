"""Deterministic agent-routing and guardrail tests (no LLM / network calls)."""
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
    req = Request(message="check this company's valuation and research report", user_id="u", conv_id="c",
                  intent=IntentCategory.VALUATION, entities={"metric": ["PE"]})
    scores = o._domain_scores(req)
    assert scores[AgentType.RESEARCH] == max(scores.values())


def test_domain_scores_compliance():
    o = _orch()
    req = Request(message="my risk assessment is R2, can I enable margin trading", user_id="u", conv_id="c",
                  intent=IntentCategory.SUITABILITY, entities={"risk_level": ["R2"]})
    scores = o._domain_scores(req)
    assert scores[AgentType.COMPLIANCE] == max(scores.values())


def test_intent_routing_map():
    assert AgentOrchestrator._INTENT_ROUTING[IntentCategory.VALUATION] == AgentType.RESEARCH
    assert AgentOrchestrator._INTENT_ROUTING[IntentCategory.SUITABILITY] == AgentType.COMPLIANCE
    assert AgentOrchestrator._INTENT_ROUTING[IntentCategory.HUMAN_HANDOFF] == AgentType.ESCALATION


# ── Investment-advice guardrail (T3) ─────────────────────────────────────────

def test_guardrail_not_triggered_for_info():
    o = _orch()
    req = Request(message="what is the fee rate of a CSI 300 ETF", user_id="u", conv_id="c",
                  intent=IntentCategory.PRODUCT_INFO)
    assert o._needs_guardrail(req) is False


def test_guardrail_detects_advice_english():
    o = _orch()
    for msg in ["recommend me a stock that will double", "should I buy this now",
                "will it go up tomorrow", "place a buy order for me"]:
        req = Request(message=msg, user_id="u", conv_id="c", intent=IntentCategory.ADVICE_REQUEST)
        assert o._needs_guardrail(req) is True, msg


def test_guardrail_keyword_without_intent():
    # Even when the caller supplies a non-advice intent, a keyword hit must still trip the guardrail.
    o = _orch()
    req = Request(message="which liquor stock should I buy right now", user_id="u", conv_id="c",
                  intent=IntentCategory.COMPARISON)
    assert o._needs_guardrail(req) is True


def test_guardrail_response_english():
    o = _orch()
    req = Request(message="recommend a stock that will double", user_id="u", conv_id="c",
                  intent=IntentCategory.ADVICE_REQUEST)
    res = o._guardrail_response(req, 0.0)
    assert res.escalated is True
    assert res.agent_type == AgentType.ESCALATION
    assert "does not constitute investment advice" in res.response
    assert "guardrail" in res.routing_reason


# ── Clarification path ───────────────────────────────────────────────────────

def test_clarification_result_is_constructible():
    """
    The clarification branch short-circuits before any LLM call, so nothing else in the
    suite reaches it. It once passed a `routing_confidence=` kwarg that no longer existed
    on OrchestratorResult, which made every low-confidence OTHER question a 500.
    """
    import asyncio
    o = _orch()
    req = Request(message="hmm what about that thing we discussed", user_id="u", conv_id="c",
                  intent=IntentCategory.OTHER, intent_confidence=0.2)
    assert o._needs_clarification(req) is True

    res = asyncio.run(o.run(req))
    assert res.escalated is False
    assert res.primary_agent == AgentType.MARKET
    assert "clarify" in res.routing_reason
    # No agent scored this turn, so there is no domain score to report.
    assert res.routing_score is None


def test_guardrail_keyword_needs_word_boundaries():
    """
    A bare substring test escalated "What is a small investment account?" as an advice
    request, because "all in" sits inside "sm(all in)vestment". Guardrail false positives
    are paid for by the user: they are refused an answer they were entitled to.

    The keywords that remain deliberately broad ("recommend", "guaranteed") are a separate,
    accepted trade-off — for investment advice, over-refusing beats under-refusing.
    """
    o = _orch()
    for msg in ["What is a small investment account?",
                "How do I make a small initial deposit?",
                "What is the installment schedule?"]:
        req = Request(message=msg, user_id="u", conv_id="c", intent=IntentCategory.PRODUCT_INFO)
        assert o._needs_guardrail(req) is False, msg

    for msg in ["I want to go all in on this",
                "recommend me a stock that will double",
                "which to buy, A or B?"]:
        req = Request(message=msg, user_id="u", conv_id="c", intent=IntentCategory.PRODUCT_INFO)
        assert o._needs_guardrail(req) is True, msg
