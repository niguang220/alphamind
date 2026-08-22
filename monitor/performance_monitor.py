"""
Highlight: monitoring agent online performance with Monitor.

Core problem: how to monitor agents' online performance?

This module's answers:
  1. Live collection -- every N seconds, pull the latest stats from the Orchestrator and ToolManager
  2. Anomaly detection -- Z-score statistics automatically catch sudden metric shifts
  3. Routing feedback -- write agent success rate/latency back to the Orchestrator,
     whose _best_agent() then dynamically adjusts routing weights
  4. Suggestions -- rule-based, actionable improvement suggestions (not empty talk)
  5. Alerts -- log on threshold breach + optional webhook
"""
import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import httpx
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

class Severity(Enum):
    INFO     = "info"
    WARNING  = "warning"
    ERROR    = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity:    Severity
    metric:      str
    message:     str
    value:       float
    threshold:   float
    ts:          str = field(default_factory=lambda: datetime.now().isoformat())
    resolved:    bool = False


@dataclass
class Suggestion:
    """An actionable improvement suggestion."""
    title:       str
    detail:      str
    action:      str    # concrete steps
    priority:    int    # 1-10


# ── Anomaly detection ─────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Sliding-window Z-score anomaly detection.

    Z-score = |value - mean| / stddev
    Flagged as an anomaly when it exceeds `sensitivity` standard deviations.
    """

    def __init__(self, window: int = 60, sensitivity: float = 2.5):
        self._window      = window
        self._sensitivity = sensitivity
        self._history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """Record a data point; return anomaly info if anomalous, else None."""
        buf = self._history[metric]
        buf.append(value)

        if len(buf) < self._window // 2:
            return None  # not enough data, skip detection

        mean  = statistics.mean(buf)
        stdev = statistics.stdev(buf) if len(buf) > 1 else 0.0
        if stdev == 0:
            return None

        z = abs(value - mean) / stdev
        if z > self._sensitivity:
            return {
                "metric":   metric,
                "value":    value,
                "mean":     mean,
                "z_score":  round(z, 2),
                "severity": "high" if z > self._sensitivity * 1.5 else "medium",
            }
        return None


# ── Performance monitor ───────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Agent online-performance monitoring.

    Loop with the Orchestrator:
      Monitor collects -> an agent's success rate drops ->
      that agent's routing_score in Orchestrator.get_stats() decreases ->
      _best_agent() routes around it automatically

    That is the closed loop of "monitoring online performance with Monitor".
    """

    # alert thresholds
    THRESHOLDS = {
        "agent_success_rate":  (0.90, Severity.ERROR,   "less_than"),
        "tool_success_rate":   (0.95, Severity.WARNING,  "less_than"),
        "agent_avg_ms":        (3000, Severity.WARNING,  "greater_than"),
        "tool_avg_ms":         (5000, Severity.ERROR,    "greater_than"),
    }

    def __init__(
        self,
        orchestrator,
        tool_manager,
        interval_s:       float = 10.0,
        webhook_url:      Optional[str] = None,
        prometheus_port:  Optional[int] = None,   # None = do not start
    ):
        self._orchestrator = orchestrator
        self._tool_manager = tool_manager
        self._interval     = interval_s
        self._webhook      = webhook_url
        self._detector     = AnomalyDetector()

        self._alerts:      List[Alert]      = []
        self._suggestions: List[Suggestion] = []
        self._active       = False
        self._task:        Optional[asyncio.Task] = None

        # Prometheus metrics (optional)
        self._prom: Dict[str, Any] = {}
        if prometheus_port:
            self._setup_prometheus(prometheus_port)

    def _setup_prometheus(self, port: int) -> None:
        self._prom = {
            "agent_success_rate": Gauge("agent_success_rate", "Agent success rate", ["agent"]),
            "agent_latency_ms":   Histogram("agent_latency_ms", "Agent latency", ["agent"]),
            "tool_success_rate":  Gauge("tool_success_rate", "Tool success rate", ["tool"]),
            "requests_total":     Counter("requests_total", "Total requests"),
        }
        start_http_server(port)
        logger.info(f"Prometheus started: :{port}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._task   = asyncio.create_task(self._loop())
        logger.info(f"Monitor started, collection interval {self._interval}s")

    async def stop(self) -> None:
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Collection loop ───────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._active:
            try:
                await self._collect()
            except Exception as ex:
                logger.error(f"Monitor collection error: {ex}")
            await asyncio.sleep(self._interval)

    async def _collect(self) -> None:
        """
        Collect live agent and tool stats, detect anomalies, generate suggestions.

        Key: the stats read here are updated live by the Orchestrator/ToolManager while
        handling requests, so Monitor needs no extra instrumentation.
        """
        agent_stats = self._orchestrator.get_stats()
        tool_stats  = self._tool_manager.get_stats()
        routing_penalties: Dict[str, float] = {}

        # ── Agent metrics ─────────────────────────────────────────────────────
        for agent_key, s in agent_stats.items():
            sr  = s["success_rate"]
            ms  = s["avg_ms"]

            # anomaly detection
            for metric, value in [("agent_success_rate", sr), ("agent_avg_ms", ms)]:
                anomaly = self._detector.record(f"{metric}:{agent_key}", value)
                if anomaly:
                    logger.warning(f"anomaly [{agent_key}] {metric}={value:.3f} z={anomaly['z_score']}")

            # threshold alert
            self._check_threshold("agent_success_rate", sr, agent_key)
            self._check_threshold("agent_avg_ms", ms, agent_key)

            # Prometheus
            if "agent_success_rate" in self._prom:
                self._prom["agent_success_rate"].labels(agent=agent_key).set(sr)
                self._prom["agent_latency_ms"].labels(agent=agent_key).observe(ms)

            routing_penalties[agent_key] = self._routing_penalty(sr, ms)

        # ── Tool metrics ──────────────────────────────────────────────────────
        for tool_name, s in tool_stats.items():
            sr = s["success_rate"]
            ms = s["avg_latency_ms"]
            cf = s["consecutive_fails"]

            self._check_threshold("tool_success_rate", sr, tool_name)
            self._check_threshold("tool_avg_ms", ms, tool_name)

            if "tool_success_rate" in self._prom:
                self._prom["tool_success_rate"].labels(tool=tool_name).set(sr)

            # consecutive failures -> generate a concrete suggestion
            if cf >= 3:
                self._add_suggestion(Suggestion(
                    title=f"Tool {tool_name} failed {cf} times in a row",
                    detail=f"success rate {sr:.1%}, avg latency {ms:.0f}ms, circuit: {s['circuit_state']}",
                    action="1. Check the tool's dependencies\n2. Review error logs\n3. Consider a longer timeout or a fallback",
                    priority=9,
                ))

        # ── Routing suggestions ───────────────────────────────────────────────
        updater = getattr(self._orchestrator, "update_routing_penalties", None)
        if updater:
            updater(routing_penalties)
        self._generate_routing_suggestions(agent_stats)

    @staticmethod
    def _routing_penalty(success_rate: float, avg_ms: float) -> float:
        """Convert online performance into a 0-0.9 routing penalty factor."""
        penalty = 0.0
        if success_rate < 0.90:
            penalty += min(0.5, (0.90 - success_rate) * 2)
        if avg_ms > 3000:
            penalty += min(0.4, (avg_ms - 3000) / 10000)
        return min(penalty, 0.9)

    def _check_threshold(self, metric: str, value: float, label: str) -> None:
        if metric not in self.THRESHOLDS:
            return
        threshold, severity, operator = self.THRESHOLDS[metric]
        triggered = (operator == "less_than" and value < threshold) or \
                    (operator == "greater_than" and value > threshold)
        if triggered:
            alert = Alert(
                severity=severity,
                metric=f"{metric}:{label}",
                message=f"{label} {metric} = {value:.3f}, threshold {threshold}",
                value=value,
                threshold=threshold,
            )
            self._alerts.append(alert)
            logger.warning(f"[{severity.value.upper()}] {alert.message}")
            # send the webhook asynchronously (do not block the collection loop)
            if self._webhook:
                asyncio.create_task(self._send_webhook(alert))

    def _generate_routing_suggestions(self, agent_stats: Dict[str, Any]) -> None:
        """
        Generate routing suggestions from agents' online performance.
        This embodies the Monitor -> Orchestrator feedback loop.
        """
        for agent_key, s in agent_stats.items():
            if s["success_rate"] < 0.85 and s["total"] > 10:
                self._add_suggestion(Suggestion(
                    title=f"Agent {agent_key} has low success rate",
                    detail=f"success rate {s['success_rate']:.1%}, routing score {s['routing_score']:.3f}",
                    action=(
                        "Orchestrator's _best_agent() has already lowered this agent's routing weight.\n"
                        "Suggestions: 1. Check whether the system_prompt needs tuning\n"
                        "             2. Check whether the problem complexity exceeds the agent's ability\n"
                        "             3. Consider adding more instances of this agent type"
                    ),
                    priority=8,
                ))

    def _add_suggestion(self, s: Suggestion) -> None:
        # dedupe: do not add duplicate titles
        if not any(x.title == s.title for x in self._suggestions):
            self._suggestions.append(s)
            logger.info(f"suggestion [P{s.priority}]: {s.title}")

    async def _send_webhook(self, alert: Alert) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                await c.post(self._webhook, json=asdict(alert))  # type: ignore
        except Exception as ex:
            logger.error(f"webhook send failed: {ex}")

    # ── Query API ─────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return the current monitoring summary for the API layer."""
        return {
            "agent_stats":   self._orchestrator.get_stats(),
            "tool_stats":    self._tool_manager.get_stats(),
            "active_alerts": [asdict(a) for a in self._alerts if not a.resolved][-10:],
            "suggestions":   [
                {"title": s.title, "action": s.action, "priority": s.priority}
                for s in sorted(self._suggestions, key=lambda x: -x.priority)[:5]
            ],
        }
