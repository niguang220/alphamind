"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 MarketAgent

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
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


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    MARKET     = "market"      # 行情与信息
    RESEARCH   = "research"    # 投研与分析
    COMPLIANCE = "compliance"  # 合规与适当性
    ESCALATION = "escalation"  # 人工投顾升级


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
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
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
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
    escalate:    bool  = False   # 是否需要升级


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
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
    routing_confidence: float = 0.0


@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

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
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[结构化实体]\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._build_system_prompt(req),
            messages=messages,
        )
        return extract_text_content(resp.content)

    def _build_system_prompt(self, req: Request) -> str:
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "人工投顾", "投资顾问", "无法处理", "escalate"]
        return any(kw in content for kw in keywords)


class MarketAgent(BaseAgent):
    agent_type    = AgentType.MARKET
    system_prompt = (
        "你是 AlphaMind 的行情与信息助手。客观、中立地提供行情、指数、个股/ETF 产品信息和术语解释。"
        "只做信息陈述，不预测涨跌、不给买卖建议。涉及具体买卖决策时，提示这属于投资决策并建议咨询持牌投顾。"
        "如信息可能有时效性，提醒用户以实时行情为准。"
    )


class ResearchAgent(BaseAgent):
    agent_type    = AgentType.RESEARCH
    system_prompt = (
        "你是 AlphaMind 的投研与分析助手。专注：研报检索与摘要、财报/基本面解读、估值与财务指标、"
        "量化概念（因子/回测/夏普/最大回撤）。"
        "只做客观分析，同时呈现多空两面与风险，不下买卖结论、不荐股、不预测点位。"
        "引用数据时说明口径与时效，并提醒历史数据不代表未来表现。"
    )


class ComplianceAgent(BaseAgent):
    agent_type    = AgentType.COMPLIANCE
    system_prompt = (
        "你是 AlphaMind 的合规与适当性助手。专注：投资者适当性与风险等级（R1-R5）、风险揭示、开户/账户、"
        "出入金与银证转账、交易规则与费率、对账单。"
        "严格遵守合规底线：不荐股、不承诺收益、不代客操作、不索要账户密码。"
        "当用户风险等级与产品风险不匹配，或涉及高风险产品时，明确提示并建议转人工投顾核验。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 MarketAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
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
        # market 组及其余意图 → MARKET（默认）
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

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.MARKET:     [MarketAgent(client, model, skill_manager)],
            AgentType.RESEARCH:   [ResearchAgent(client, model, skill_manager)],
            AgentType.COMPLIANCE: [ComplianceAgent(client, model, skill_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence

        # 2. 投资建议护栏:命中荐股/择时/保收益/代客请求 → 拦截并升级,不进入正常回答
        if self._needs_guardrail(req):
            logger.info(f"请求 {req.request_id} 命中投资建议护栏")
            return self._guardrail_response(req, (time.monotonic() - t0) * 1000)

        if self._needs_clarification(req):
            return OrchestratorResult(
                request_id=req.request_id,
                response="我还不能确定您的问题类别。请问是行情/产品信息、投研分析（研报/估值/财报），还是账户/适当性/交易规则？",
                agent_type=AgentType.MARKET,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.MARKET],
                primary_agent=AgentType.MARKET,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
            )

        # 复杂问题自动并行协作，例如同一句同时涉及登录故障和扣款/退款。
        decision = self._route_decision(req)
        if decision.multi_agent:
            return await self.run_parallel(req, decision)

        # 2. 执行主 Agent（含降级）
        response = await self._execute(req, decision.primary_agent)

        # 4. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION,
            IntentCategory.HUMAN_HANDOFF,
        ):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知人工客服

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
            routing_confidence=decision.confidence,
        )

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        t0 = time.monotonic()
        agent_types = decision.agent_types
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：主 Agent 在前，辅助 Agent 在后。
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                role = "主处理" if r.agent_type == decision.primary_agent else "辅助处理"
                parts.append(f"[{r.agent_type.value} - {role}]\n{r.content}")

        combined = "\n\n".join(parts) if parts else "抱歉，所有 Agent 均处理失败。"
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
            routing_confidence=decision.confidence,
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.MARKET

    def _route_decision(self, req: Request) -> RoutingDecision:
        """
        结构化路由决策。

        先处理紧急/转人工，再用领域分数决定主 Agent 和辅助 Agent。
        这样可以表达“主处理 + 辅助诊断”，避免关键词命中后无主次地拼接。
        """
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="紧急度为 CRITICAL，触发升级路由",
                confidence=1.0,
            )

        if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason=f"意图为 {req.intent.value if req.intent else 'unknown'}，触发升级路由",
                confidence=max(req.intent_confidence, 0.8),
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
                reason="无可用专属 Agent，降级到 MarketAgent",
                confidence=0.1,
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
            confidence=round(min(primary_score, 1.0), 3),
        )

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
        """按意图、关键词和实体为各领域 Agent 打分。"""
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

        research_kws = ["研报", "财报", "年报", "季报", "估值", "市盈率", "pe", "pb", "roe", "因子", "回测", "夏普", "回撤", "基本面", "营收", "净利润"]
        compliance_kws = ["开户", "销户", "适当性", "风险等级", "风险测评", "风险揭示", "银证转账", "出金", "入金", "对账单", "交割单", "佣金", "费率", "r1", "r2", "r3", "r4", "r5"]
        market_kws = ["行情", "指数", "股价", "etf", "基金", "点位", "术语", "什么意思", "涨跌停", "交易时间"]

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
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"登录报错且被重复扣款"需要技术和账单 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        research_kws = ["研报", "财报", "估值", "市盈率", "因子", "回测", "基本面", "跟踪误差", "对比"]
        compliance_kws = ["适当性", "风险等级", "风险测评", "风险揭示", "开户", "银证转账", "出金", "入金", "费率", "能买"]

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

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    # ── 投资建议护栏 ──────────────────────────────────────────────────────────
    # 命中即拦截:不给出个股买卖建议、择时判断或收益承诺,改为风险揭示 + 转人工投顾。
    _GUARDRAIL_KEYWORDS = [
        "推荐", "买哪", "该买", "该不该买", "会涨", "会不会涨", "能涨到", "涨不涨",
        "保本", "保收益", "稳赚", "包赚", "帮我下单", "帮我买", "帮我卖", "代客",
        "全仓", "梭哈", "抄底", "能不能买", "值不值得买", "要不要买",
    ]

    def _needs_guardrail(self, req: Request) -> bool:
        """是否命中投资建议护栏:意图为 advice_request,或消息含荐股/择时/保收益/代客等关键词。"""
        if req.intent == IntentCategory.ADVICE_REQUEST:
            return True
        msg = (req.message or "").lower()
        return any(kw in msg for kw in self._GUARDRAIL_KEYWORDS)

    def _guardrail_response(self, req: Request, elapsed_ms: float) -> OrchestratorResult:
        """护栏安全响应:合规拒答 + 风险揭示 + 升级,不进入领域 Agent 生成。"""
        text = (
            "AlphaMind 提供证券投研信息与投资者教育，不构成投资建议，也不能代您做出买卖决策或下单。\n"
            "关于个股/产品的买卖判断，请注意以下风险提示：\n"
            "1. 证券投资有风险，历史表现不代表未来收益，任何人都无法保证收益或保本；\n"
            "2. 是否适合某产品，取决于您的风险承受能力（适当性/风险等级 R1-R5）与投资目标；\n"
            "3. 具体买卖决策建议咨询持牌投资顾问。\n"
            "我可以帮您：解读该标的的基本面/估值/研报要点、说明产品风险与交易规则，或为您转接人工投顾。"
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
            routing_confidence=1.0,
        )

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 MarketAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.MARKET)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.MARKET,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 MarketAgent
        if not response.success and agent_type != AgentType.MARKET:
            logger.warning(f"{agent_type.value} 失败，降级到 MarketAgent")
            fallback = self._best_agent(AgentType.MARKET)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

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
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
