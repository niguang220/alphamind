"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "AlphaMind RAG 知识库"},
        )

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（证券投研场景常见问题）。"""
        default_docs = [
            {
                "title": "常见投资术语与指标词典",
                "content": (
                    "常见投资术语与财务指标说明。"
                    "市盈率（PE）：股价除以每股收益，衡量估值高低，可与行业或历史分位比较。"
                    "市净率（PB）：股价除以每股净资产，常用于重资产或金融行业估值。"
                    "净资产收益率（ROE）：净利润除以净资产，衡量股东回报能力。"
                    "每股收益（EPS）：净利润除以总股本。"
                    "夏普比率：单位风险（波动率）所获得的超额收益，越高越好。"
                    "最大回撤：区间内从最高点到最低点的最大跌幅，衡量下行风险。"
                    "Beta：衡量个股相对大盘的波动敏感度，Beta>1 波动大于大盘。"
                    "波动率：收益率的标准差，反映价格波动剧烈程度。"
                    "换手率：成交量占流通股本的比例，反映交投活跃度。"
                ),
            },
            {
                "title": "ETF 产品说明",
                "content": (
                    "ETF（交易型开放式指数基金）说明。"
                    "宽基 ETF 跟踪沪深300、中证500、创业板等综合指数；行业 ETF 跟踪某一行业；跨境 ETF 跟踪境外指数。"
                    "管理费与托管费合计通常在 0.2%-0.6%/年，宽基 ETF 费率普遍较低。"
                    "跟踪误差衡量 ETF 净值与标的指数的偏离，越小说明复制越紧密。"
                    "场内 ETF 可像股票一样 T+1 买卖；申赎以一篮子成分股进行，可能产生折溢价。"
                    "折溢价率＝（市价−净值）/净值，长期大幅溢价需警惕。"
                ),
            },
            {
                "title": "投资者适当性与风险等级",
                "content": (
                    "投资者适当性管理说明。"
                    "个人投资者风险承受能力通常分为 C1（保守）到 C5（激进）五级。"
                    "产品风险等级分为 R1（低）到 R5（高）五级。"
                    "适当性原则要求把合适的产品卖给合适的投资者，客户风险等级应不低于产品风险等级。"
                    "例如 R2 稳健型客户通常不匹配 R5 高风险产品（如两融、期权、部分私募）。"
                    "首次购买高风险产品或超风险等级购买需进行风险提示与双录（录音录像）。"
                    "风险测评结果一般有效期约两年，到期或情况变化需重新测评。"
                ),
            },
            {
                "title": "证券交易规则",
                "content": (
                    "A 股证券交易规则说明。"
                    "结算制度为 T+1：当日买入的股票次一交易日才能卖出，资金 T+1 可取。"
                    "涨跌停：主板 ±10%，创业板/科创板 ±20%，ST 股票 ±5%。"
                    "交易时段：集合竞价 9:15-9:25，连续竞价 9:30-11:30 与 13:00-15:00。"
                    "交易费用包含佣金（双向，通常不超过成交额 0.03% 左右且有最低 5 元）、"
                    "印花税（卖出单边收取 0.05%）、过户费（沪深按成交额 0.001% 左右）。"
                ),
            },
            {
                "title": "风险揭示书要点",
                "content": (
                    "证券投资风险揭示要点。"
                    "市场风险：价格受宏观、行业、公司基本面与情绪影响而波动，可能导致本金亏损。"
                    "流动性风险：部分标的成交清淡，可能无法及时以合理价格买卖。"
                    "信用风险：债券等发行主体可能违约。"
                    "杠杆风险：融资融券、期权等带杠杆，亏损可能被放大甚至爆仓强平。"
                    "汇率风险：跨境投资受汇率波动影响。"
                    "历史业绩不代表未来表现，任何机构和个人都不得承诺保本或保收益。"
                ),
            },
            {
                "title": "开户与账户管理",
                "content": (
                    "证券开户与账户管理说明。"
                    "开户需完成实名认证（KYC）、风险测评并签署相关协议。"
                    "资金实行第三方存管，证券账户与绑定的银行卡通过银证转账划转资金。"
                    "银证转账一般在交易日的规定时段办理，出入金到账时间以券商与银行为准。"
                    "销户需先清空持仓与资金并了结未了事项。"
                    "账户安全：不要向任何人泄露账户密码与验证码，正规客服与投顾不会索要密码。"
                ),
            },
            {
                "title": "财报与基本面解读指引",
                "content": (
                    "财报与基本面解读指引。"
                    "三大报表：利润表反映盈利能力，资产负债表反映财务状况，现金流量表反映现金流。"
                    "营业收入与净利润反映规模与盈利，需关注同比/环比增速与稳定性。"
                    "毛利率＝（营收−营业成本）/营收，反映产品竞争力。"
                    "经营性现金流应与净利润相互印证，长期净利高而现金流差需警惕。"
                    "分析应结合行业景气、可比公司与历史区间，避免只看单一指标下结论。"
                ),
            },
            {
                "title": "合规红线与投资者教育",
                "content": (
                    "合规红线与投资者教育。"
                    "本平台仅提供证券投研信息、数据检索与投资者教育，不构成投资建议。"
                    "不荐股、不预测点位、不承诺收益、不代客理财或代客下单。"
                    "涉及具体买卖决策请咨询持牌投资顾问，并结合自身风险承受能力独立判断。"
                    "理性投资，坚持长期视角，警惕‘保本高收益’‘内幕消息’‘稳赚不赔’等违规话术与诈骗。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
