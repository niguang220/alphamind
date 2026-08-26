"""
Does multi-agent collaboration cost latency, or only tokens?

`run_parallel` dispatches the primary and supporting agents through `asyncio.gather`. If that
is working, end-to-end wall-clock should track the slowest single call rather than the sum —
adding an agent should cost tokens, not user-visible latency.

That is the claim this measures. It is deliberately not a claim about `asyncio.gather` itself
(which is defined to behave that way); it is a check that nothing in the orchestrator path —
a shared lock, a sequential await, a client that serialises — has quietly removed the
concurrency. Regressions of exactly that kind are easy to introduce and invisible in unit
tests.

Run:  .venv/bin/python benchmarks/parallel_dispatch.py --repeats 5
Writes benchmarks/results/parallel_dispatch.json
"""
import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "bench-key")

from agents.agent_orchestrator import AgentType, AgentOrchestrator, Request  # noqa: E402
from core.intent_recognizer import IntentCategory  # noqa: E402


class SleepingAgent:
    """Stands in for a domain agent at a fixed, known latency, so the harness measures the
    orchestrator's dispatch and not the model provider's variance."""

    def __init__(self, agent_type, delay_s):
        from agents.agent_orchestrator import AgentStats
        self.agent_type = agent_type
        self.delay_s = delay_s
        self.calls = 0
        # _best_agent ranks instances by stats.routing_score(); without this the stub is
        # silently skipped and the fallback path returns in microseconds.
        self.stats = AgentStats()

    async def handle(self, req):
        from agents.agent_orchestrator import AgentResponse
        self.calls += 1
        t0 = time.monotonic()
        await asyncio.sleep(self.delay_s)
        return AgentResponse(
            agent_type=self.agent_type,
            content=f"[{self.agent_type.value}] stub",
            success=True,
            latency_ms=(time.monotonic() - t0) * 1000,
        )


async def one_run(delays):
    o = AgentOrchestrator(api_key="bench-key", base_url="https://example.invalid")
    agents = {t: SleepingAgent(t, d) for t, d in delays.items()}
    for t, a in agents.items():
        o._pool[t] = [a]

    req = Request(
        message="for 600519.SH with a P/E of 30 and ROE of 25 and a max drawdown of 40%, "
                "what risk rating applies and can an R2 investor hold it",
        user_id="bench", conv_id="bench",
        intent=IntentCategory.SUITABILITY,
        entities={"ticker": ["600519.SH"], "metric": ["P/E", "ROE"],
                  "risk_level": ["R2"], "percentage": ["40%"]},
    )
    decision = o._route_decision(req)
    if not decision.multi_agent:
        raise SystemExit(f"benchmark question no longer triggers collaboration: {decision.reason}")

    t0 = time.monotonic()
    await o.run_parallel(req, decision)
    wall = time.monotonic() - t0
    dispatched = [t for t in decision.agent_types]
    return wall, dispatched


async def main(repeats):
    delays = {AgentType.COMPLIANCE: 0.40, AgentType.RESEARCH: 0.25, AgentType.MARKET: 0.10}
    walls, dispatched = [], None
    for _ in range(repeats):
        w, dispatched = await one_run(delays)
        walls.append(w)

    used = [delays[t] for t in dispatched]
    slowest, serial = max(used), sum(used)
    median = statistics.median(walls)

    result = {
        "repeats": repeats,
        "agents_dispatched": [t.value for t in dispatched],
        "per_agent_delay_s": {t.value: delays[t] for t in dispatched},
        "slowest_single_s": round(slowest, 4),
        "serial_would_be_s": round(serial, 4),
        "wall_clock_median_s": round(median, 4),
        "wall_over_slowest": round(median / slowest, 3),
        "wall_over_serial": round(median / serial, 3),
        "walls_s": [round(w, 4) for w in walls],
    }
    out = pathlib.Path(__file__).parent / "results" / "parallel_dispatch.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nwall-clock is {result['wall_over_slowest']}x the slowest single agent "
          f"({result['wall_over_serial']}x what serial dispatch would cost)")
    print(f"written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    asyncio.run(main(a.repeats))
