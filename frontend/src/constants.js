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
    color: 'var(--color-violet)',
    blurb: 'Hands the question to a licensed human advisor.',
  },
}

export const agentMeta = (key) =>
  AGENTS[key] ?? { label: key || 'unknown', full: key || 'unknown', color: 'var(--color-ink-mute)', blurb: '' }

/**
 * The four stages of a /chat request, each timed on the server and returned in
 * `timings`. Nothing here is estimated: the bars show what the request actually spent.
 */
export const STAGES = [
  { id: 'memory', label: 'Memory', detail: 'recall the thread', key: 'memory' },
  { id: 'intent', label: 'Intent', detail: 'classify the question', key: 'intent' },
  { id: 'knowledge', label: 'Knowledge', detail: 'retrieve sources', key: 'knowledge' },
  { id: 'orchestration', label: 'Agent', detail: 'screen, route, answer', key: 'orchestration' },
]

export const SAMPLES = [
  {
    text: 'should I buy into this dip',
    tag: 'Guardrail',
    note: 'Screened out before any agent runs',
    color: 'var(--color-brass)',
  },
  {
    text: 'what is the difference between a broad-based index ETF and a sector ETF',
    tag: 'Research',
    note: 'Cites the product guide',
    color: 'var(--color-teal)',
  },
  {
    text: 'what does a risk rating of R3 allow me to buy',
    tag: 'Compliance',
    note: 'Matches R3 against product risk grades',
    color: 'var(--color-brass)',
  },
  {
    text: 'how is the maximum drawdown of a fund calculated',
    tag: 'Market',
    note: 'Straight from the glossary',
    color: 'var(--color-azure)',
  },
]

// Last recorded evaluation run (POST /eval/run, deepseek-chat). Static on purpose: a live
// run is a ~59s batch, not an interaction.
//
// Shown as counts, not rates. These cases are the built-in evaluation set — small, and
// scored by the same model family that produced the answers — so a rate to two decimals
// would imply a precision the sample size does not support. The demo prompts above are
// deliberately NOT drawn from this set.
export const EVAL_SNAPSHOT = {
  ranAt: '2026-08-23',
  model: 'deepseek-chat',
  headline: [
    { key: 'guardrail intercepted', pass: 1, of: 1 },
    { key: 'intent correct', pass: 12, of: 12 },
    { key: 'dialog passed', pass: 6, of: 6 },
  ],
  quality: [
    { key: 'relevance', value: 0.9812 },
    { key: 'accuracy', value: 0.975 },
    { key: 'helpfulness', value: 0.9062 },
    { key: 'completeness', value: 0.875 },
  ],
  caveat: 'Quality scores are LLM-judged — directional, not independent. Guardrail interception is a deterministic assertion.',
}
