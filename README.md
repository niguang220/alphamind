# AlphaMind — Securities Investment-Research Assistant

> AlphaMind is a **multi-agent runtime** for securities investment research: securities research Q&A, investor education, and report/data retrieval.
> **It does not constitute investment advice; it does not recommend stocks, promise returns, or trade on your behalf.** Requests for buy/sell advice, market timing, guaranteed returns, or trading on the user's behalf are intercepted by an **investment-advice guardrail** and escalated to a human advisor.
>
> Bilingual: replies follow the language of the user's question (Chinese or English). Chinese docs: [README.zh.md](README.zh.md).

Core flow:

```text
User request
  -> FastAPI /chat
  -> MemoryManager reads Redis working memory + ChromaDB episodic memory + user profile
  -> IntentRecognizer classifies intent via 3-way fusion (LLM + Embedding + keywords)
  -> ⚠️ Investment-advice guardrail: stock picks / timing / guaranteed returns / trade-for-me -> intercept + risk disclosure + escalate
  -> Intent-gated RAG (query rewrite -> parallel recall -> dedup -> LLM rerank)
  -> AgentOrchestrator routes to Market / Research / Compliance agents (primary + supporting)
  -> inject dynamic Skills -> LLM generates the reply
  -> write to Redis, async update ChromaDB user profile
  -> Monitor online routing penalty + LLM-as-Judge evaluation loop
```

## 1. Capabilities

| Capability | Description |
|------------|-------------|
| Fine-grained intent recognition | ~20 securities intents in three groups (market / research / compliance) + guardrail; LLM + Embedding + keyword weighted voting |
| **Investment-advice guardrail** | intent-gated guardrail: stock-pick / timing / guaranteed-return / trade-for-me requests are intercepted, returning a risk disclosure and escalating to a human advisor |
| Intent-gated RAG | only business-information intents query the ChromaDB knowledge base; query rewrite, parallel recall, dedup, LLM rerank |
| Multi-agent routing | Market & Information / Research & Analysis / Compliance & Suitability; outputs primary + supporting, routing_reason, confidence |
| Layered memory | Redis working memory (24h TTL) + ChromaDB episodic memory + user profile; auto-compress over threshold |
| Dynamic Skills | market / research / compliance guidelines, injected by agent type + keywords, hot-reloadable |
| Tool reliability | parameter validation, TTL cache, timeout, circuit breaker, and fallback for the knowledge-search tool |
| Observability & evaluation | Monitor writes back routing_penalty from success rate/latency; LLM-as-Judge four-dimension scoring + guardrail hit rate + regression detection |

## 2. Agents

| Agent | Responsibility |
|-------|----------------|
| **MarketAgent** (Market & Information) | quotes / indices / ETF & stock product info / term explanations; state facts, no price predictions |
| **ResearchAgent** (Research & Analysis) | research retrieval, financials/fundamentals, valuation and quant concepts (factor/backtest/Sharpe/drawdown); interpret only, no buy/sell conclusions |
| **ComplianceAgent** (Compliance & Suitability) | risk ratings R1–R5, suitability matching, risk disclosure, account opening, trading rules & fees |
| **Escalation** (human advisor) | on guardrail hits, complaints, or serious suitability mismatch, transfer to a human advisor |

## 3. Intents (~20, 3 groups + guardrail)

- **Market group → Market**: `market_quote` `product_info` `term_explain` `trading_rule`
- **Research group → Research**: `research_report` `fundamental` `valuation` `comparison` `quant_concept`
- **Compliance group → Compliance**: `account` `funding` `suitability` `risk_disclosure` `statement`
- **Flow / guardrail**: ⚠️`advice_request` (stock picks / timing / guaranteed returns / trade-for-me), `complaint` `human_handoff` `escalation` `greeting` `feedback` `other`

## 4. Quick start

### 4.1 Requirements

- Docker + Docker Compose
- An Anthropic API key, or an Anthropic-compatible third-party key (e.g., DeepSeek)

Configure `.env` (minimum):

```env
ANTHROPIC_API_KEY=your_api_key
# Third-party compatible endpoint example:
# ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
# ANTHROPIC_MODEL=deepseek-v4-pro
```

### 4.2 Full stack via Docker Compose

```bash
docker compose up -d --build
docker compose logs -f alphamind      # container/network names keep the alphamind_ prefix
curl http://localhost:8000/health
# Swagger: http://localhost:8000/docs
```

Starts: AlphaMind API (8000), Nginx (80), ChromaDB (8001), Redis (6379), Prometheus (9090).

### 4.3 CLI

```bash
docker compose run --rm alphamind python api/main.py --cli
```

### 4.4 Run tests locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q     # deterministic unit tests: intent / routing / guardrail / skills (Chinese & English)
```

## 5. API overview

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | health check |
| `POST` | `/chat` | main chat: memory -> intent -> guardrail -> gated RAG -> routing -> reply |
| `POST` | `/search` | RAG retrieval pipeline (query rewrite / parallel recall / rerank) |
| `GET` | `/monitor` | agent/tool metrics, alerts, suggestions |
| `GET` `POST` | `/skills` `/skills/reload` | view / hot-reload dynamic Skills |
| `POST` | `/knowledge/add` `/knowledge/upload` | import knowledge-base documents |
| `GET` | `/knowledge/stats` | knowledge-base chunk count |
| `POST` | `/eval/run` | end-to-end evaluation (includes guardrail_hit_rate) |

### 5.1 /chat examples

Information question (hits knowledge base + the right agent):

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "What are the fee and tracking error of the CSI 300 ETF?",
  "user_id": "u1", "conv_id": "c1"
}'
```

Suitability question (routes to Compliance):

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "My risk assessment is R2, can I open margin trading?", "user_id": "u1", "conv_id": "c1"
}'
```

⚠️ Guardrail example (stock-pick request -> refuse + risk disclosure + escalate):

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "Recommend me a stock that can double, ideally rising tomorrow", "user_id": "u1", "conv_id": "c1"
}'
# Returns escalated=true; response contains "does not constitute investment advice" + risk notice + suggests a human advisor
```

## 6. Knowledge base

`mcp/knowledge_base.py` uses the ChromaDB `knowledge_base` collection. On first start, if empty, it auto-imports 8 default securities documents: terms & metrics glossary, ETF product guide, investor suitability & risk ratings, securities trading rules, risk disclosure essentials, account opening & management, financials/fundamentals reading guide, and compliance red lines & investor education. Demo upload files are under `data/demo_docs/`.

## 7. Memory & evaluation

- **Layered memory**: Redis `wm:{user}:{conv}` (24h TTL, compress at 15 messages, keep the latest 5); ChromaDB `episodic` (history summaries) and `user_profile`.
- **Evaluation**: `POST /eval/run` actually calls the Orchestrator to generate replies, scores them with LLM-as-Judge on four dimensions (relevance / accuracy / completeness / helpfulness), and reports the **guardrail hit rate `guardrail_hit_rate`**, intent Accuracy/Macro-F1, and regression detection.

## 8. Tech stack

Python 3.12 · FastAPI · Anthropic SDK (DeepSeek-compatible) · Redis · ChromaDB · Prometheus · Docker Compose · pytest

## 9. Compliance notice

AlphaMind provides securities research information, investor education, and data retrieval only. It **does not constitute investment advice; it does not recommend stocks, predict price levels, promise returns, or trade on your behalf.** For specific investment decisions, consult a licensed investment advisor and judge independently based on your own risk tolerance.
