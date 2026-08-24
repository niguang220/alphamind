export const AGENTS = {
  market: {
    label: 'Market',
    full: 'Market & Information',
    color: 'var(--color-azure)',
    blurb: 'Quotes, indices, product facts, trading rules.',
  },
  research: {
    label: 'Research',
    full: 'Research & Analysis',
    color: 'var(--color-teal)',
    blurb: 'Reports, fundamentals, valuation, quant concepts.',
  },
  compliance: {
    label: 'Compliance',
    full: 'Compliance & Suitability',
    color: 'var(--color-brass)',
    blurb: 'Suitability, risk ratings, account and funding rules.',
  },
  escalation: {
    label: 'Escalation',
    full: 'Human Advisor Escalation',
    color: 'var(--color-ember)',
    blurb: 'Routed away from automated answers.',
  },
}

export const agentMeta = (key) =>
  AGENTS[key] ?? { label: key || 'unknown', full: key || 'unknown', color: 'var(--color-ink-mute)', blurb: '' }

/**
 * The four stages are the real ones in AgentOrchestrator.run():
 *   intent recognition -> guardrail -> routing decision -> agent execution.
 * The per-stage durations below are indicative pacing, not backend telemetry —
 * the backend returns one response, and the only measured number we show is the
 * end-to-end latency_ms it reports.
 */
export const STAGES = [
  { id: 'intent', label: 'Intent', detail: 'read the question', at: 0 },
  { id: 'guardrail', label: 'Guardrail', detail: 'advice check', at: 700 },
  { id: 'routing', label: 'Routing', detail: 'match an agent', at: 1200 },
  { id: 'generation', label: 'Generation', detail: 'write the answer', at: 1800 },
]

export const SAMPLES = [
  {
    text: 'recommend a stock that will double',
    tag: 'Guardrail',
    note: 'Stopped by the guardrail',
    color: 'var(--color-brass)',
  },
  {
    text: "is the CSI 300 ETF's valuation expensive? show P/E and ROE",
    tag: 'Research',
    note: 'Pulls from the knowledge base',
    color: 'var(--color-teal)',
  },
  {
    text: 'my risk assessment is R2, can I open margin trading?',
    tag: 'Compliance',
    note: 'Reads your risk rating',
    color: 'var(--color-brass)',
  },
  {
    text: 'what are the A-share trading hours and price limits',
    tag: 'Market',
    note: 'Straight market facts',
    color: 'var(--color-azure)',
  },
]

// Snapshot of the last full evaluation run (POST /eval/run, deepseek-chat).
// Kept static on purpose: a live run takes ~59s, which is not a demo interaction.
export const EVAL_SNAPSHOT = {
  ranAt: '2026-08-23',
  model: 'deepseek-chat',
  wallClock: '58.7s',
  headline: [
    { key: 'guardrail_hit_rate', value: 1.0 },
    { key: 'intent_accuracy', value: 1.0 },
    { key: 'pass_rate', value: 1.0 },
  ],
  quality: [
    { key: 'relevance', value: 0.9812 },
    { key: 'accuracy', value: 0.975 },
    { key: 'helpfulness', value: 0.9062 },
    { key: 'completeness', value: 0.875 },
  ],
  cases: '9 cases · 12 intent + 6 dialog',
}
