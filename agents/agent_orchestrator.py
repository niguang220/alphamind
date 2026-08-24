"""
Highlight: multi-agent routing and orchestration.

Core problem: how to do routing across multiple agents?

Routing strategy (three layers):
  1. Intent routing -- map IntentCategory directly to a specialized agent
  2. Performance routing -- when a type has several instances, pick the highest success / lowest latency one
  3. Fallback routing -- when a specialized agent is unavailable, fall back to MarketAgent

Parallel collaboration:
  - Complex questions can be dispatched to several agents at once
  - The Orchestrator merges the results

Escalation:
  - When an agent's confidence is below threshold, escalate to a higher agent or a human
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

class AgentType(Enum):
    MARKET     = "market"      # Market & Information
    RESEARCH   = "research"    # Research & Analysis
    COMPLIANCE = "compliance"  # Compliance & Suitability
    ESCALATION = "escalation"  # human advisor escalation


@dataclass
class AgentStats:
    """Agent runtime stats, used by Monitor and routing decisions."""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """Routing score: agents with higher success rate and lower latency score higher."""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # whether escalation is needed


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # formatted context from MemoryManager
    history:     Optional[List[Dict[str, str]]] = None  # conversation history, passed to intent recognition
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    routing_reason: str = ""
    # Domain score of the agent that took the turn: an additive sum of intent, keyword and
    # entity bonuses, so it is unbounded and is not a probability. None when the route was
    # decided by a rule rather than by scoring.
    routing_score: Optional[float] = None


@dataclass
class RoutingDecision:
    """Structured routing decision for one request."""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    # Additive domain score, unbounded; None when a rule decided the route.
    score: Optional[float] = None

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# ── Base agent ───────────────────────────────────────────────────────────────

class BaseAgent:
    """Base class for all agents; wraps LLM calls and stats."""

    agent_type: AgentType
    system_prompt: str

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} failed: {ex}")
            fallback = "Sorry, something went wrong while handling your request. Please try again later."
            return AgentResponse(
                agent_type=self.agent_type,
                content=fallback,
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[Context]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "Understood, I have reviewed the context."})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[Structured entities]\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": "Understood, I will use these entities."})
        messages.append({"role": "user", "content": _clean(req.message) + self._LANG_REMINDER})

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=self._build_system_prompt(req),
            messages=messages,
        )
        return extract_text_content(resp.content)

    # Replies follow the user's language while prompts and knowledge stay English.
    #
    # This takes two instructions working together, and measurably so: over five runs of an
    # English question about the CSI 300, the system directive alone still answered in Chinese
    # 3 times, and a bare reminder on the user turn alone answered in Chinese 5 times. Together
    # they were 0 for 20 across both languages. The system half has to name the failure mode —
    # a Chinese-trained model reaches for Chinese the moment the subject is a Chinese market —
    # and the user-turn half supplies the recency that a system prompt cannot.
    _LANG_DIRECTIVE = (
        "\n\nWrite your entire reply in the same language as the user's latest message. "
        "If the user writes in English, answer in English even when the subject is a Chinese "
        "market, index, company or regulation — translate any Chinese source material instead "
        "of switching language. If the user writes in Chinese, answer in Chinese."
    )
    _LANG_REMINDER = "\n\n[Reply in the same language as this message.]"

    def _build_system_prompt(self, req: Request) -> str:
        """Splice dynamically loaded Skills into the system prompt so business rules apply per request."""
        base = self.system_prompt + self._LANG_DIRECTIVE
        if self._skill_manager is None:
            return base
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base
        return f"{base}\n\n[Dynamic Skills]\n{skill_prompt}"

    def _needs_escalation(self, content: str) -> bool:
        """Detect whether the agent suggests escalation (simple keyword check)."""
        keywords = ["escalate", "human advisor", "cannot handle"]
        return any(kw in content for kw in keywords)


class MarketAgent(BaseAgent):
    agent_type    = AgentType.MARKET
    system_prompt = (
        "You are AlphaMind's Market & Information assistant. Provide market quotes, indices, "
        "stock/ETF product information and term explanations objectively and neutrally. "
        "State facts only; do NOT predict price movements or give buy/sell advice. "
        "When a question involves an actual buy/sell decision, note that this is an investment decision "
        "and suggest consulting a licensed advisor. Remind users that quotes are time-sensitive and to rely on real-time data."
    )


class ResearchAgent(BaseAgent):
    agent_type    = AgentType.RESEARCH
    system_prompt = (
        "You are AlphaMind's Research & Analysis assistant. Focus on: research report retrieval and summaries, "
        "financial statement / fundamentals interpretation, valuation and financial metrics, and quantitative concepts "
        "(factors, backtesting, Sharpe ratio, maximum drawdown). "
        "Give objective analysis presenting both bullish and bearish sides and the key risks; "
        "do NOT draw buy/sell conclusions, recommend stocks, or predict price targets. "
        "State the basis and timeliness of any data, and remind users that past performance does not indicate future results."
    )


class ComplianceAgent(BaseAgent):
    agent_type    = AgentType.COMPLIANCE
    system_prompt = (
        "You are AlphaMind's Compliance & Suitability assistant. Focus on: investor suitability and risk ratings (R1-R5), "
        "risk disclosure, account opening/management, deposits/withdrawals and bank-securities transfers, "
        "trading rules and fees, and statements. "
        "Strictly follow compliance limits: do NOT recommend stocks, promise returns, trade on the client's behalf, or ask for account passwords. "
        "When the user's risk rating does not match a product's risk, or a high-risk product is involved, "
        "clearly flag it and suggest transferring to a human advisor for verification."
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Multi-agent orchestrator.

    Routing logic (three layers):
      1. intent -> agent-type mapping
      2. pick the best instance by routing_score() when a type has several
      3. fall back to MarketAgent when the specialized agent fails
    """

    # static intent -> agent-type mapping (routing table)
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.RESEARCH_REPORT: AgentType.RESEARCH,
        IntentCategory.FUNDAMENTAL:     AgentType.RESEARCH,
        IntentCategory.VALUATION:       AgentType.RESEARCH,
        IntentCategory.COMPARISON:      AgentType.RESEARCH,
        IntentCategory.QUANT_CONCEPT:   AgentType.RESEARCH,
        IntentCategory.ACCOUNT:         AgentType.COMPLIANCE,
        IntentCategory.FUNDING:         AgentType.COMPLIANCE,
        IntentCategory.SUITABILITY:     AgentType.COMPLIANCE,
        IntentCategory.RISK_DISCLOSURE: AgentType.COMPLIANCE,
        IntentCategory.STATEMENT:       AgentType.COMPLIANCE,
        IntentCategory.ESCALATION:      AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF:   AgentType.ESCALATION,
        # market group and all other intents -> MARKET (default)
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager

        # agent pool: each type may have several instances (horizontal scaling)
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.MARKET:     [MarketAgent(client, model, skill_manager)],
            AgentType.RESEARCH:   [ResearchAgent(client, model, skill_manager)],
            AgentType.COMPLIANCE: [ComplianceAgent(client, model, skill_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """Update the SkillManager reference, for runtime reload or test replacement."""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """Expose intent recognition so the API layer can decide on RAG etc. beforehand."""
        return await self._intent_recognizer.recognize(message, history=history)

    # ── Main entry ────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        Full pipeline for one request:
          intent recognition -> route to agent -> execute -> check escalation -> return result
        """
        t0 = time.monotonic()

        # 1. intent recognition (skip if the caller already provided it)
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence

        # 2. investment-advice guardrail: intercept & escalate stock-pick/timing/guaranteed-return/trade-for-me requests
        if self._needs_guardrail(req):
            logger.info(f"request {req.request_id} hit the investment-advice guardrail")
            return self._guardrail_response(req, (time.monotonic() - t0) * 1000)

        if self._needs_clarification(req):
            clarify = ("I'm not sure which category your question falls into. Is it about market / product information, "
                       "research analysis (reports / valuation / financials), or account / suitability / trading rules?")
            return OrchestratorResult(
                request_id=req.request_id,
                response=clarify,
                agent_type=AgentType.MARKET,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.MARKET],
                primary_agent=AgentType.MARKET,
                routing_reason="low-confidence OTHER intent, ask user to clarify",
                routing_confidence=req.intent_confidence,
            )

        # Auto parallel collaboration for complex questions, e.g. one message about research/valuation and suitability/risk-rating.
        decision = self._route_decision(req)
        if decision.multi_agent:
            return await self.run_parallel(req, decision)

        # 2. execute the primary agent (with fallback)
        response = await self._execute(req, decision.primary_agent)

        # 4. escalation check
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION,
            IntentCategory.HUMAN_HANDOFF,
        ):
            escalated = True
            logger.warning(f"request {req.request_id} triggered escalation: urgency={req.urgency}")
            # production: create a ticket / hand off to a human advisor / compliance here

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            routing_reason=decision.reason,
            routing_score=decision.score,
        )

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        """
        Dispatch to several agents in parallel and merge the results.
        For complex questions spanning multiple domains.
        """
        t0 = time.monotonic()
        agent_types = decision.agent_types
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # merge: primary agent first, supporting agents after.
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                role = "primary" if r.agent_type == decision.primary_agent else "supporting"
                parts.append(f"[{r.agent_type.value} - {role}]\n{r.content}")

        if parts:
            combined = "\n\n".join(parts)
        else:
            combined = "Sorry, all agents failed to process the request."
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[
                r.agent_type for r in responses
                if isinstance(r, AgentResponse) and r.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            routing_reason=decision.reason,
            routing_score=decision.score,
        )

    # ── Routing logic ─────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        Three-layer routing decision:
          1. intent mapping
          2. urgency override (CRITICAL escalates directly)
          3. default MARKET
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # use the target type if it has an available instance, otherwise fall back
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.MARKET

    def _route_decision(self, req: Request) -> RoutingDecision:
        """
        Structured routing decision.

        Handle urgency / human handoff first, then use domain scores to pick primary and supporting agents.
        This expresses "primary handling + supporting diagnosis" instead of concatenating on keyword hits.
        """
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="urgency=CRITICAL, escalation route",
                score=None,
            )

        if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason=f"intent={req.intent.value if req.intent else 'unknown'}, escalation route",
                score=None,
            )

        scores = self._domain_scores(req)
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.MARKET or self._pool.get(agent_type)
        }
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.MARKET,
                reason="no specialized agent available, fallback to MarketAgent",
                score=0.1,
            )

        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]
        supporting_agents = [
            agent_type
            for agent_type, score in ordered[1:]
            if agent_type != AgentType.MARKET and score >= 0.45 and score >= primary_score * 0.55
        ]

        reason = self._routing_reason(req, available_scores, primary_agent, supporting_agents)
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            score=round(primary_score, 3),
        )

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
        """Score each domain agent by intent, keywords and entities."""
        msg = req.message.lower()
        scores = {
            AgentType.MARKET: 0.1,
            AgentType.RESEARCH: 0.0,
            AgentType.COMPLIANCE: 0.0,
        }

        if req.intent in (
            IntentCategory.MARKET_QUOTE,
            IntentCategory.PRODUCT_INFO,
            IntentCategory.TERM_EXPLAIN,
            IntentCategory.TRADING_RULE,
            IntentCategory.GREETING,
            IntentCategory.FEEDBACK,
            IntentCategory.OTHER,
        ):
            scores[AgentType.MARKET] += 0.55

        if req.intent in (
            IntentCategory.RESEARCH_REPORT,
            IntentCategory.FUNDAMENTAL,
            IntentCategory.VALUATION,
            IntentCategory.COMPARISON,
            IntentCategory.QUANT_CONCEPT,
        ):
            scores[AgentType.RESEARCH] += 0.75

        if req.intent in (
            IntentCategory.ACCOUNT,
            IntentCategory.FUNDING,
            IntentCategory.SUITABILITY,
            IntentCategory.RISK_DISCLOSURE,
            IntentCategory.STATEMENT,
            IntentCategory.COMPLAINT,
        ):
            scores[AgentType.COMPLIANCE] += 0.75

        research_kws = ["research report", "annual report", "quarterly report", "earnings", "valuation",
                        "p/e", "pe ratio", "p/b", "roe", "factor", "backtest", "sharpe", "drawdown",
                        "fundamentals", "revenue", "net profit"]
        compliance_kws = ["open an account", "close an account", "suitability", "risk level", "risk rating",
                          "risk assessment", "risk disclosure", "bank transfer", "withdraw", "deposit",
                          "statement", "trade confirmation", "commission", "fee rate",
                          "r1", "r2", "r3", "r4", "r5"]
        market_kws = ["quote", "index", "share price", "etf", "fund", "index level", "terminology",
                      "what does it mean", "limit up", "limit down", "trading hours"]

        research_hits = sum(1 for kw in research_kws if kw in msg)
        compliance_hits = sum(1 for kw in compliance_kws if kw in msg)
        market_hits = sum(1 for kw in market_kws if kw in msg)

        scores[AgentType.RESEARCH] += min(0.45, research_hits * 0.18)
        scores[AgentType.COMPLIANCE] += min(0.45, compliance_hits * 0.18)
        scores[AgentType.MARKET] += min(0.35, market_hits * 0.12)

        entities = req.entities or {}
        if entities.get("metric"):
            scores[AgentType.RESEARCH] += 0.2
        if entities.get("risk_level"):
            scores[AgentType.COMPLIANCE] += 0.2
        if entities.get("ticker"):
            scores[AgentType.RESEARCH] += 0.1

        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    @staticmethod
    def _routing_reason(
        req: Request,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        score_text = ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return (
            f"intent={intent}, group={req.intent_group or 'unknown'}, "
            f"primary={primary_agent.value}, supporting={support_text}, scores=[{score_text}]"
        )

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        Decide whether several agents should collaborate in parallel.

        Intent recognition usually returns a single primary intent; here domain keywords
        additionally detect compound questions that need two agents at once.
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        research_kws = ["research report", "earnings", "valuation", "p/e", "pe ratio", "factor",
                        "backtest", "fundamentals", "tracking error", "compare"]
        compliance_kws = ["suitability", "risk level", "risk rating", "risk assessment", "risk disclosure",
                          "open an account", "bank transfer", "withdraw", "deposit", "fee rate", "can i buy"]

        if req.intent in (
            IntentCategory.RESEARCH_REPORT,
            IntentCategory.FUNDAMENTAL,
            IntentCategory.VALUATION,
            IntentCategory.COMPARISON,
            IntentCategory.QUANT_CONCEPT,
        ) or any(kw in msg for kw in research_kws):
            targets.append(AgentType.RESEARCH)
        if req.intent in (
            IntentCategory.ACCOUNT,
            IntentCategory.FUNDING,
            IntentCategory.SUITABILITY,
            IntentCategory.RISK_DISCLOSURE,
            IntentCategory.STATEMENT,
        ) or any(kw in msg for kw in compliance_kws):
            targets.append(AgentType.COMPLIANCE)

        # dedupe preserving order, and only return agent types that currently have instances.
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """When confidence is low and intent is unclear, ask first to avoid misrouting."""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    # ── Investment-advice guardrail ───────────────────────────────────────────
    # On a hit, intercept: give no individual buy/sell advice, timing calls or return promises; return a risk disclosure + human advisor.
    _GUARDRAIL_KEYWORDS = [
        "recommend", "should i buy", "should i sell", "will it go up", "will it rise",
        "guaranteed", "buy for me", "sell for me", "place a buy order", "place an order",
        "trade for me", "which to buy", "what to buy", "worth buying", "all in",
        "double my money", "price target", "tell me what to buy",
    ]

    def _needs_guardrail(self, req: Request) -> bool:
        """Whether the investment-advice guardrail is hit: intent is advice_request, or the message contains stock-pick/timing/guaranteed-return/trade-for-me keywords."""
        if req.intent == IntentCategory.ADVICE_REQUEST:
            return True
        msg = (req.message or "").lower()
        return any(kw in msg for kw in self._GUARDRAIL_KEYWORDS)

    def _guardrail_response(self, req: Request, elapsed_ms: float) -> OrchestratorResult:
        """Guardrail-safe response: compliant refusal + risk disclosure + escalation, bypassing domain agents."""
        text = (
            "AlphaMind provides securities research information and investor education. "
            "This does not constitute investment advice, and I cannot make buy/sell decisions or place orders for you.\n"
            "Regarding buy/sell judgments on a specific security or product, please note:\n"
            "1. Securities investing carries risk; past performance does not indicate future results, and no one can guarantee returns or principal;\n"
            "2. Whether a product suits you depends on your risk tolerance (suitability / risk rating R1-R5) and objectives;\n"
            "3. For specific buy/sell decisions, please consult a licensed investment advisor.\n"
            "I can help you: interpret the fundamentals / valuation / research highlights of the security, "
            "explain product risks and trading rules, or connect you to a human advisor."
        )
        return OrchestratorResult(
            request_id=req.request_id,
            response=text,
            agent_type=AgentType.ESCALATION,
            intent=req.intent,
            escalated=True,
            latency_ms=elapsed_ms,
            agent_types=[AgentType.ESCALATION],
            primary_agent=AgentType.ESCALATION,
            routing_reason=f"guardrail=advice_request,intent={req.intent.value if req.intent else 'unknown'}",
            routing_score=None,  # guardrail is a rule, not a score
        )

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        Performance routing: pick the highest routing_score() among same-type agents.
        This is the core of "dynamically adjusting routing based on online performance".
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """Execute an agent; fall back to MarketAgent on failure."""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.MARKET)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.MARKET,
                content="Service is temporarily unavailable, please try again later.",
                success=False,
            )

        response = await agent.handle(req)

        # fall back to MarketAgent when the specialized agent fails
        if not response.success and agent_type != AgentType.MARKET:
            logger.warning(f"{agent_type.value} failed, falling back to MarketAgent")
            fallback = self._best_agent(AgentType.MARKET)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── Stats (read by Monitor) ───────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        Receive Monitor's online-performance feedback and adjust routing penalties dynamically.

        The keys of penalties use the agent keys from get_stats(), e.g. research_0.
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
