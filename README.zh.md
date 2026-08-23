# AlphaMind 证券投研智能咨询助手

> AlphaMind 是一个面向证券投研场景的**多 Agent 编排运行时**:提供证券投研信息问答、投资者教育和研报/数据检索。
> **它不构成投资建议、不荐股、不承诺收益、不代客操作**;对索要买卖建议、择时、保收益或代客的请求,会由**投资建议护栏**拦截并转人工投顾。

核心链路:

```text
用户请求
  -> FastAPI /chat
  -> MemoryManager 读取 Redis 工作记忆 + ChromaDB 情景记忆 + 用户画像
  -> IntentRecognizer 三路融合识别意图(LLM + Embedding + 关键词)
  -> ⚠️ 投资建议护栏:命中荐股/择时/保收益/代客 → 拦截 + 风险揭示 + 升级
  -> 按意图门控 RAG(查询改写 → 并行召回 → 去重 → LLM 重排)
  -> AgentOrchestrator 路由到 Market / Research / Compliance Agent(主+辅)
  -> 注入动态 Skills → LLM 生成回复
  -> 写入 Redis,异步更新 ChromaDB 用户画像
  -> Monitor 在线降权 + LLM-as-Judge 评测闭环
```

## 1. 能力总览

| 能力 | 说明 |
|------|------|
| 细粒度意图识别 | ~20 类证券意图,分行情/投研/合规三组 + 护栏,LLM + Embedding + 关键词三路加权投票 |
| **投资建议护栏** | intent-gated guardrail:荐股/择时/保收益/代客请求被拦截,返回风险揭示并升级人工投顾 |
| 意图门控 RAG | 仅业务信息类意图检索 ChromaDB 知识库;查询改写、并行召回、去重、LLM 重排 |
| 多 Agent 路由 | 行情与信息 / 投研与分析 / 合规与适当性,输出 primary + supporting、routing_reason、confidence |
| 三级记忆 | Redis 工作记忆(24h TTL)+ ChromaDB 情景记忆 + 用户画像,超阈值自动压缩 |
| 动态 Skills | 行情/投研/合规三类规范,按 Agent 类型 + 关键词注入,支持热加载 |
| 工具治理 | 知识库检索的参数校验、TTL 缓存、超时、熔断、fallback 降级 |
| 观测与评测 | Monitor 按成功率/延迟写回 routing_penalty;LLM-as-Judge 四维评分 + 护栏拦截准确率 + 回归检测 |

## 2. Agent 角色

| Agent | 职责 |
|-------|------|
| **MarketAgent**(行情与信息) | 行情/指数/ETF与个股产品信息、术语解释;只陈述信息,不预测涨跌 |
| **ResearchAgent**(投研与分析) | 研报检索、财报/基本面解读、估值与量化概念(因子/回测/夏普/回撤);只客观解读,不下买卖结论 |
| **ComplianceAgent**(合规与适当性) | 风险等级 R1–R5、适当性匹配、风险揭示、开户/账户、交易规则与费率 |
| **Escalation**(人工投顾升级) | 护栏命中、投诉、适当性严重不匹配时转人工投顾 |

## 3. 意图体系(~20 类,3 组 + 护栏)

- **行情信息组 → Market**:`market_quote` `product_info` `term_explain` `trading_rule`
- **投研分析组 → Research**:`research_report` `fundamental` `valuation` `comparison` `quant_concept`
- **合规适当性组 → Compliance**:`account` `funding` `suitability` `risk_disclosure` `statement`
- **护栏/流程**:⚠️`advice_request`(荐股/择时/保收益/代客)、`complaint` `human_handoff` `escalation` `greeting` `feedback` `other`

## 4. 快速开始

### 4.1 环境

- Docker + Docker Compose
- Anthropic API Key,或兼容 Anthropic 协议的第三方 Key(如 DeepSeek)

配置 `.env`(最少):

```env
ANTHROPIC_API_KEY=your_api_key
# 兼容第三方接口示例
# ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
# 用 deepseek-chat。不要用 deepseek-v4-pro：它是推理模型，thinking block 会吃光
# max_tokens 预算，导致 text block 为空、回复空白。
# ANTHROPIC_MODEL=deepseek-chat
```

### 4.2 Docker Compose 全栈部署

```bash
docker compose up -d --build
docker compose logs -f alphamind
curl http://localhost:8000/health
# Swagger: http://localhost:8000/docs
```

启动服务:AlphaMind API(8000)、Nginx(80)、ChromaDB(8001)、Redis(6379)、Prometheus(9090)。

### 4.3 CLI 交互

```bash
docker compose run --rm alphamind python api/main.py --cli
```

### 4.4 本地运行测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q     # 意图/路由/护栏/Skills 确定性单测
```

## 5. 接口总览

| 方法 | 路径 | 作用 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/chat` | 主对话:记忆 → 意图 → 护栏 → 门控 RAG → 路由 → 回复 |
| `POST` | `/search` | RAG 检索优化链路(查询改写/并行召回/重排) |
| `GET` | `/monitor` | Agent/工具指标、告警、优化建议 |
| `GET` `POST` | `/skills` `/skills/reload` | 查看/热加载动态 Skills |
| `POST` | `/knowledge/add` `/knowledge/upload` | 导入知识库文档 |
| `GET` | `/knowledge/stats` | 知识库片段数 |
| `POST` | `/eval/run` | 端到端评测(含 guardrail_hit_rate) |

### 5.1 /chat 示例

信息类问题(命中知识库 + 对应 Agent):

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "沪深300ETF的费率和跟踪误差大概是多少？",
  "user_id": "u1", "conv_id": "c1"
}'
```

适当性问题(路由到 Compliance):

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "我风险测评是R2，想开通两融可以吗？", "user_id": "u1", "conv_id": "c1"
}'
```

⚠️ 护栏拦截示例(荐股请求 → 拒答 + 风险揭示 + 升级):

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "message": "帮我推荐一只能翻倍的股票，最好明天就涨", "user_id": "u1", "conv_id": "c1"
}'
# 返回 escalated=true,response 含“不构成投资建议”+ 风险提示 + 建议转人工投顾
```

## 6. 知识库

`mcp/knowledge_base.py` 使用 ChromaDB `knowledge_base` collection。首次启动若为空,自动导入 8 篇默认证券文档:术语指标词典、ETF 产品说明、投资者适当性与风险等级、证券交易规则、风险揭示书要点、开户与账户管理、财报基本面解读、合规红线与投资者教育。演示上传文件见 `data/demo_docs/`。

## 7. 记忆与评测

- **三级记忆**:Redis `wm:{user}:{conv}`(24h TTL,达 15 条压缩,保留最近 5 条);ChromaDB `episodic`(历史摘要)、`user_profile`(用户画像)。
- **评测**:`POST /eval/run` 真实调用 Orchestrator 生成回复,LLM-as-Judge 四维评分(相关性/准确性/完整性/有用性),并统计**护栏拦截准确率 `guardrail_hit_rate`**、意图 Accuracy/Macro-F1 与回归检测。

## 8. 技术栈

Python 3.12 · FastAPI · Anthropic SDK(兼容 DeepSeek)· Redis · ChromaDB · Prometheus · Docker Compose · pytest

## 9. 合规声明

AlphaMind 仅提供证券投研信息、投资者教育与数据检索,**不构成投资建议,不荐股、不预测点位、不承诺收益、不代客操作**。具体投资决策请咨询持牌投资顾问并结合自身风险承受能力独立判断。
