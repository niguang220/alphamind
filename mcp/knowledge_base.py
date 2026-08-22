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
            metadata={"description": "AlphaMind RAG knowledge base"},
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
        """Load default knowledge base documents (common securities-research topics)."""
        default_docs = [
            {
                "title": "Common Investment Terms & Metrics Glossary",
                "content": (
                    "Glossary of common investment terms and financial metrics. "
                    "P/E ratio: price divided by earnings per share; gauges valuation, compared across peers or history. "
                    "P/B ratio: price divided by net assets per share; often used for asset-heavy or financial firms. "
                    "ROE: net profit divided by equity; measures shareholder return. "
                    "EPS: net profit divided by total shares. "
                    "Sharpe ratio: excess return per unit of risk (volatility); higher is better. "
                    "Maximum drawdown: the largest peak-to-trough decline over a period; measures downside risk. "
                    "Beta: sensitivity of a stock relative to the market; Beta above 1 means larger swings than the market. "
                    "Volatility: standard deviation of returns. Turnover: volume as a share of float, showing trading activity."
                ),
            },
            {
                "title": "ETF Product Guide",
                "content": (
                    "About ETFs (Exchange-Traded Funds). "
                    "Broad-based ETFs track composite indices such as CSI 300, CSI 500 or ChiNext; sector ETFs track an industry; "
                    "cross-border ETFs track overseas indices. "
                    "Total management and custody fees are usually 0.2 to 0.6 percent per year, with broad-based ETFs generally cheaper. "
                    "Tracking error measures how closely the ETF NAV follows its index; smaller is better. "
                    "On-exchange ETFs trade like stocks under T+1; creation and redemption use a basket of constituents and may show a premium or discount. "
                    "Premium/discount equals (market price minus NAV) divided by NAV; a persistent large premium warrants caution."
                ),
            },
            {
                "title": "Investor Suitability & Risk Ratings",
                "content": (
                    "About investor suitability management. "
                    "An individual investor risk tolerance is usually graded from C1 (conservative) to C5 (aggressive). "
                    "Product risk is graded from R1 (low) to R5 (high). "
                    "The suitability principle requires matching the right product to the right investor; the client risk grade should be no lower than the product risk grade. "
                    "For example, an R2 conservative client generally does not match an R5 high-risk product such as margin trading, options or some private funds. "
                    "Buying a high-risk product for the first time or above one risk grade requires risk warnings and audio/video recording. "
                    "A risk assessment is typically valid for about two years and must be redone when it expires or circumstances change."
                ),
            },
            {
                "title": "Securities Trading Rules",
                "content": (
                    "A-share securities trading rules. "
                    "Settlement is T+1: shares bought today can be sold the next trading day; cash is available T+1. "
                    "Price limits: Main Board plus or minus 10 percent, ChiNext and STAR Market plus or minus 20 percent, ST stocks plus or minus 5 percent. "
                    "Trading hours: call auction 9:15 to 9:25, continuous auction 9:30 to 11:30 and 13:00 to 15:00. "
                    "Fees include commission (both sides, usually up to about 0.03 percent of turnover with a minimum around 5 CNY), "
                    "stamp duty (0.05 percent, charged on the sell side only), and a transfer fee (about 0.001 percent of turnover)."
                ),
            },
            {
                "title": "Risk Disclosure Essentials",
                "content": (
                    "Key risk disclosures for securities investing. "
                    "Market risk: prices swing with macro, sector, fundamentals and sentiment, and may cause principal loss. "
                    "Liquidity risk: some instruments trade thinly and may not be bought or sold in time at a fair price. "
                    "Credit risk: bond issuers may default. "
                    "Leverage risk: margin trading and options carry leverage; losses can be amplified and may lead to forced liquidation. "
                    "Currency risk: cross-border investing is exposed to FX moves. "
                    "Past performance does not indicate future results; no institution or individual may promise principal protection or returns."
                ),
            },
            {
                "title": "Account Opening & Management",
                "content": (
                    "Securities account opening and management. "
                    "Opening an account requires identity verification (KYC), a risk assessment, and signing the relevant agreements. "
                    "Funds use third-party depository; money moves between the securities account and the linked bank card via bank-securities transfer. "
                    "Bank-securities transfers are processed during designated windows on trading days; deposit and withdrawal timing depends on the broker and bank. "
                    "Closing an account requires clearing positions and cash and settling any outstanding items. "
                    "Account security: never disclose your account password or verification codes; legitimate staff and advisors will never ask for them."
                ),
            },
            {
                "title": "Financial Statement & Fundamentals Reading Guide",
                "content": (
                    "Guide to reading financial statements and fundamentals. "
                    "Three statements: the income statement shows profitability, the balance sheet shows financial position, and the cash flow statement shows cash flows. "
                    "Revenue and net profit show scale and profitability; watch year-over-year and quarter-over-quarter growth and its stability. "
                    "Gross margin equals (revenue minus cost of sales) divided by revenue, reflecting product competitiveness. "
                    "Operating cash flow should corroborate net profit; high net profit with weak cash flow warrants caution. "
                    "Analysis should combine industry cycle, comparable companies and historical ranges, not a single metric in isolation."
                ),
            },
            {
                "title": "Compliance Red Lines & Investor Education",
                "content": (
                    "Compliance red lines and investor education. "
                    "This platform only provides securities research information, data retrieval and investor education; it does not constitute investment advice. "
                    "It does not recommend stocks, predict price levels, promise returns, or manage or trade on a client behalf. "
                    "For specific buy or sell decisions, consult a licensed investment advisor and judge independently based on your own risk tolerance. "
                    "Invest rationally with a long-term view; beware of illegal or fraudulent pitches such as guaranteed high returns, inside information or sure profits."
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
