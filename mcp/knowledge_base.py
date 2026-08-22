"""
RAG knowledge base -- real retrieval on top of ChromaDB.

Features:
  1. Document import: chunk text and store it in ChromaDB (embeddings generated automatically)
  2. Semantic search: retrieve the most relevant chunks for a query
  3. MCP integration: acts as the real handler for the knowledge_search tool

ChromaDB's role here:
  - memory/ uses it for conversation memory (episodic memory + user profile)
  - here it stores knowledge-base documents (RAG retrieval)
  They are different collections and do not interfere.
"""
import asyncio
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    RAG knowledge base backed by ChromaDB.

    ChromaDB has a built-in embedding model (all-MiniLM-L6-v2):
    add() generates vectors automatically and query() does semantic matching,
    so no separate Anthropic Embeddings API call is needed.
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # Prefer a standalone ChromaDB server (server has the embedding model; client needs no download)
        self._use_server = False
        try:
            # HttpClient also initializes ChromaDB telemetry by default; disable it to avoid posthog error logs.
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"KB ChromaDB connected: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"KB ChromaDB server unavailable, using local mode: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # With a server, do not pass embedding_function; let the server handle it
        # In local mode also omit it and use ChromaDB's default (triggers a model download)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "AlphaMind RAG knowledge base"},
        )

        # if the KB is empty, import default documents
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── Document management ───────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Batch-import documents into the knowledge base.

        documents format: [{"title": "...", "content": "..."}, ...]
        Long documents are auto-chunked (~500 chars each).
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
            # ChromaDB generates embeddings automatically
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"KB imported {len(ids)} document chunks")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """Async import; the ChromaDB client is synchronous, so run it in a thread pool."""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search: return the most relevant chunks for the query.

        ChromaDB turns the query into a vector and does cosine similarity against stored vectors.
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
                    "score":    round(1.0 - dist, 4),  # ChromaDB returns distance; convert to similarity
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Async search; the ChromaDB client is synchronous, so run it in a thread pool."""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """Async chunk count."""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP tool handler ──────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        Registered as the handler for the MCP tool.

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── Internal methods ──────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split long text by chunk_size, keeping semantic units (split on periods/newlines)."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # split by sentence, handling both Chinese (。！？) and English (. ! ?) enders and newlines
        sentences = [p for p in re.split(r"(?<=[。！？.!?])\s+|\n+", text) if p and p.strip()]
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current} {sent}" if current else sent

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
        logger.info(f"Imported default knowledge base: {len(default_docs)} documents")
