"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    # 行情信息组 → MARKET
    MARKET_QUOTE    = "market_quote"      # 行情/指数/涨跌查询
    PRODUCT_INFO    = "product_info"      # 个股/ETF/基金产品信息
    TERM_EXPLAIN    = "term_explain"      # 术语/概念解释
    TRADING_RULE    = "trading_rule"      # 交易规则
    # 投研分析组 → RESEARCH
    RESEARCH_REPORT = "research_report"   # 研报检索/摘要
    FUNDAMENTAL     = "fundamental"       # 基本面/财报解读
    VALUATION       = "valuation"         # 估值/财务指标
    COMPARISON      = "comparison"        # 个股/ETF 对比
    QUANT_CONCEPT   = "quant_concept"     # 量化/因子/回测/夏普/回撤
    # 合规适当性组 → COMPLIANCE
    ACCOUNT         = "account"           # 开户/账户管理
    FUNDING         = "funding"           # 出入金/银证转账/资金流水
    SUITABILITY     = "suitability"       # 风险测评/适当性/风险等级
    RISK_DISCLOSURE = "risk_disclosure"   # 风险揭示
    STATEMENT       = "statement"         # 对账单/交割单/税务
    # 流程 / 护栏
    ADVICE_REQUEST  = "advice_request"    # ⚠️护栏:荐股/择时/保收益/代客
    COMPLAINT       = "complaint"         # 投诉
    HUMAN_HANDOFF   = "human_handoff"     # 转人工投顾
    ESCALATION      = "escalation"        # 升级
    GREETING        = "greeting"          # 问候
    FEEDBACK        = "feedback"          # 正面反馈
    OTHER           = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.MARKET_QUOTE:    ["What's the CSI 300 index at today", "current price of the stock", "did ChiNext go up"],
    IntentCategory.PRODUCT_INFO:    ["what is the fee of this ETF", "what index does this ETF track", "top holdings of the fund"],
    IntentCategory.TERM_EXPLAIN:    ["what does P/E ratio mean", "how to understand the Sharpe ratio", "what is maximum drawdown"],
    IntentCategory.TRADING_RULE:    ["is A-share T+1 settlement", "what is the price limit", "what are the trading hours"],
    IntentCategory.RESEARCH_REPORT: ["is there a research report on this company", "find me a broker research summary", "latest sector research report"],
    IntentCategory.FUNDAMENTAL:     ["help me read this company's financial report", "its revenue and net profit", "how is the gross margin"],
    IntentCategory.VALUATION:       ["is the valuation expensive now", "what is the P/E percentile", "how to compute P/B and ROE"],
    IntentCategory.COMPARISON:      ["which of these two ETFs is better", "compare these two companies", "difference between broad-based and sector ETF"],
    IntentCategory.QUANT_CONCEPT:   ["what is the momentum factor", "how to compute backtest annualized return", "what are maximum drawdown and Sharpe"],
    IntentCategory.ACCOUNT:         ["how to open an account", "how to change my linked bank card", "how to close my account"],
    IntentCategory.FUNDING:         ["how to do a bank-securities transfer", "how long does withdrawal take", "my deposit hasn't arrived"],
    IntentCategory.SUITABILITY:     ["my risk assessment is R2", "how is investor suitability evaluated", "what can an R3 investor buy"],
    IntentCategory.RISK_DISCLOSURE: ["what are the risks of this product", "where is the risk disclosure", "what are the risks of leverage"],
    IntentCategory.STATEMENT:       ["where can I see my account statement", "how to download the trade confirmation", "export my transaction records"],
    IntentCategory.ADVICE_REQUEST:  ["recommend me a stock that can double", "which one should I buy now", "will it go up tomorrow", "place a buy order for me", "is it guaranteed to make money"],
    IntentCategory.COMPLAINT:       ["your service is terrible", "waited a long time and no one helped", "this is awful"],
    IntentCategory.HUMAN_HANDOFF:   ["transfer me to a human advisor", "I want a real person", "connect me to an investment advisor"],
    IntentCategory.ESCALATION:      ["I want to file a complaint", "get me your supervisor", "escalate this"],
    IntentCategory.GREETING:        ["hello", "hi there", "good morning"],
    IntentCategory.FEEDBACK:        ["that was very clear", "thank you so much", "great, thumbs up"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.MARKET_QUOTE, IntentCategory.PRODUCT_INFO, IntentCategory.TERM_EXPLAIN,
    IntentCategory.TRADING_RULE, IntentCategory.RESEARCH_REPORT, IntentCategory.FUNDAMENTAL,
    IntentCategory.VALUATION, IntentCategory.COMPARISON, IntentCategory.QUANT_CONCEPT,
    IntentCategory.ACCOUNT, IntentCategory.FUNDING, IntentCategory.SUITABILITY,
    IntentCategory.RISK_DISCLOSURE, IntentCategory.STATEMENT, IntentCategory.ADVICE_REQUEST,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.COMPLAINT,
    IntentCategory.ESCALATION,
    IntentCategory.GREETING,
    IntentCategory.FEEDBACK,
}

# 意图 → 分组字符串(market / research / compliance / escalation)
_INTENT_GROUPS: Dict[IntentCategory, str] = {
    IntentCategory.MARKET_QUOTE: "market",
    IntentCategory.PRODUCT_INFO: "market",
    IntentCategory.TERM_EXPLAIN: "market",
    IntentCategory.TRADING_RULE: "market",
    IntentCategory.RESEARCH_REPORT: "research",
    IntentCategory.FUNDAMENTAL: "research",
    IntentCategory.VALUATION: "research",
    IntentCategory.COMPARISON: "research",
    IntentCategory.QUANT_CONCEPT: "research",
    IntentCategory.ACCOUNT: "compliance",
    IntentCategory.FUNDING: "compliance",
    IntentCategory.SUITABILITY: "compliance",
    IntentCategory.RISK_DISCLOSURE: "compliance",
    IntentCategory.STATEMENT: "compliance",
    IntentCategory.COMPLAINT: "compliance",
    IntentCategory.ADVICE_REQUEST: "escalation",
    IntentCategory.HUMAN_HANDOFF: "escalation",
    IntentCategory.ESCALATION: "escalation",
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻", "爆仓", "强平", "被盗", "资金异常"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        # 第三方兼容 API（如 DeepSeek）通常不支持 Embedding，禁用该策略。
        # 官方 Anthropic SDK 当前没有 embeddings 资源，因此下面会使用稳定的
        # 本地字符 n-gram 向量作为轻量兜底，保证三路融合链路真实可跑。
        self._embedding_enabled = not bool(base_url)

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  message: "{t}" -> intent: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\nRecent conversation:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""You are an intent classifier for a securities investment-research assistant. Classify the user's intent based on the examples and return JSON.
Prefer the most specific business intent over a broad category.
For example: research retrieval -> research_report, valuation analysis -> valuation, risk assessment / suitability -> suitability.
If the user asks for buy/sell advice, stock picks, market timing, guaranteed returns, or to trade on their behalf, return advice_request.

Examples:
{examples}

{ctx}
User message: "{message}"

Return format (JSON only, no other text):
{{"intent": "<intent value>", "confidence": <0-1>, "reasoning": "<one short sentence>"}}

Available intents: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        # 关键词双语:保留中文,并补英文,保证中英文输入都能命中(意图识别本身以 LLM 为主,关键词为兜底)。
        specific_patterns = {
            IntentCategory.ADVICE_REQUEST: ["推荐", "买哪", "该买", "该不该买", "会涨", "会不会涨",
                "能涨到", "涨不涨", "保本", "保收益", "稳赚", "包赚", "帮我下单", "帮我买", "代客", "全仓", "梭哈", "抄底",
                "recommend", "should i buy", "should i sell", "which to buy", "what to buy", "will it go up",
                "will it rise", "guaranteed", "buy for me", "place a buy order", "worth buying", "all in", "price target"],
            IntentCategory.RESEARCH_REPORT: ["研报", "研究报告", "券商报告", "评级报告",
                "research report", "broker report", "analyst report", "rating report"],
            IntentCategory.FUNDAMENTAL: ["财报", "年报", "季报", "营收", "净利润", "毛利率", "基本面",
                "financial report", "annual report", "quarterly report", "revenue", "net profit", "gross margin", "fundamentals", "earnings"],
            IntentCategory.VALUATION: ["估值", "市盈率", "pe", "pb", "市净率", "roe", "分位",
                "valuation", "p/e", "p/b", "percentile", "market cap"],
            IntentCategory.QUANT_CONCEPT: ["因子", "回测", "夏普", "最大回撤", "波动率", "beta", "阿尔法", "alpha",
                "factor", "backtest", "sharpe", "drawdown", "volatility"],
            IntentCategory.COMPARISON: ["对比", "比较", "哪个更好", "区别", "vs",
                "compare", "comparison", "which is better", "difference"],
            IntentCategory.PRODUCT_INFO: ["etf", "基金", "跟踪误差", "成分股", "重仓", "费率", "申赎",
                "fund", "tracking error", "holdings", "expense ratio", "fee"],
            IntentCategory.TRADING_RULE: ["t+1", "涨跌停", "交易时间", "交易时段", "集合竞价", "佣金", "印花税", "过户费",
                "price limit", "trading hours", "auction", "commission", "stamp duty", "settlement"],
            IntentCategory.TERM_EXPLAIN: ["什么意思", "怎么理解", "是什么", "什么叫", "概念",
                "what does", "what is", "meaning of", "how to understand", "definition"],
            IntentCategory.MARKET_QUOTE: ["行情", "指数", "点位", "股价", "涨了", "跌了", "多少点",
                "index", "quote", "points", "went up", "went down"],
            IntentCategory.SUITABILITY: ["风险测评", "适当性", "风险等级", "r1", "r2", "r3", "r4", "r5", "能买",
                "suitability", "risk level", "risk assessment", "risk rating", "can i buy", "am i allowed"],
            IntentCategory.RISK_DISCLOSURE: ["风险揭示", "有什么风险", "风险提示", "杠杆风险",
                "risk disclosure", "what are the risks", "risk warning", "leverage risk"],
            IntentCategory.ACCOUNT: ["开户", "销户", "账户", "绑定银行卡", "三方存管",
                "open an account", "open account", "close account", "bank card", "custody"],
            IntentCategory.FUNDING: ["银证转账", "出金", "入金", "转账", "资金流水", "到账",
                "bank-securities transfer", "withdrawal", "deposit", "cash flow"],
            IntentCategory.STATEMENT: ["对账单", "交割单", "交易流水", "税务凭证",
                "account statement", "trade confirmation", "transaction record"],
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工投顾", "找人工", "投资顾问",
                "human advisor", "real person", "investment advisor", "talk to someone"],
        }
        generic_patterns = {
            IntentCategory.ESCALATION: ["投诉", "主管", "经理", "supervisor", "complaint", "manager"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "垃圾", "等了很久", "terrible", "awful", "worst"],
            IntentCategory.GREETING:   ["你好", "在吗", "早上好", "hello", "good morning"],
            IntentCategory.FEEDBACK:   ["谢谢", "感谢", "很清楚", "点赞", "满意", "thank", "helpful", "great"],
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best]
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        if best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS and pat_conf >= 0.5 and best_score < 0.8:
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        # 说明:中文字符在 Python re 中属于 \w,因此不能依赖 \b 做左边界
        # (如 "级R3" 中 级 与 R 之间无 \b)。这里改用 lookaround 保证在中文旁也能命中。
        return {
            "ticker": self._unique(re.findall(r"(?<![A-Za-z0-9])(\d{6}|[A-Z]{2,5})(?![A-Za-z0-9])", message)),
            "metric": self._unique(re.findall(
                r"(市盈率|市净率|夏普比率|最大回撤|波动率|市盈|夏普|P/E|P/B|P/S|PE|PB|ROE|ROA|EPS|Sharpe|drawdown|Beta|Alpha|阿尔法|贝塔)",
                message, re.I)),
            "risk_level": [v.upper() for v in self._unique(
                re.findall(r"(?<![A-Za-z])([RrCc][1-5])(?![0-9])", message))],
            "percentage": self._unique(re.findall(r"(\d+(?:\.\d+)?\s*%|百分之\d+(?:\.\d+)?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|万元|万|块|rmb|cny|usd|美元))", message, re.I)),
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|今年|去年|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)[:200]}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, "other")

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
