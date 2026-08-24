import { Meter } from './Bits'

/**
 * Intent is a weighted vote across three routes.
 *
 * Which routes are live, and therefore what the weights are, is decided by the backend at
 * construction: an Anthropic-compatible third-party provider exposes no embeddings API, so
 * that route is switched off and its weight is redistributed. Both facts come from the
 * response rather than being assumed here — a score of exactly 0.00 is a legitimate result
 * from a live route, and reading it as "switched off" mislabels a working system.
 */
const ROUTES = {
  llm: { label: 'LLM', note: 'reads the question semantically' },
  embedding: { label: 'Embedding', note: 'similarity against labelled examples' },
  pattern: { label: 'Pattern', note: 'keyword safety net' },
}

export default function FusionBars({ scores, confidence }) {
  const s = scores ?? {}
  // A route the backend did not run is absent from the payload; a route that ran and scored
  // nothing is present with 0.
  const live = Object.keys(ROUTES).filter((k) => k in s)
  const weights = live.includes('embedding')
    ? { llm: 0.7, embedding: 0.2, pattern: 0.1 }
    : { llm: 0.85, pattern: 0.15 }

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <div className="label">How the intent was read</div>
        <div className="tabular text-[12px] text-ink-soft">
          <span className="text-teal">{(confidence ?? 0).toFixed(2)}</span> combined
        </div>
      </div>

      <div className="mt-3 space-y-3">
        {Object.entries(ROUTES).map(([key, r], i) => {
          const off = !(key in s)
          const val = s[key] ?? 0
          return (
            <div key={key}>
              <div className="flex items-baseline justify-between gap-3">
                <span
                  className={`font-mono text-[11px] tracking-[0.08em] uppercase ${
                    off ? 'text-ink-faint' : 'text-ink-soft'
                  }`}
                >
                  {r.label}
                </span>
                <span className="tabular text-[11px] text-ink-mute">
                  {off ? '—' : val.toFixed(2)}
                  <span className="ml-2 text-ink-faint">
                    {off ? 'off' : `w ${weights[key].toFixed(2)}`}
                  </span>
                </span>
              </div>
              <div className="mt-1.5">
                <Meter value={val} muted={off} color="var(--color-teal)" delay={i * 90} />
              </div>
              <p className="mt-1.5 text-[11px] text-ink-faint">
                {off
                  ? 'This provider has no embeddings API — weight went to the other two.'
                  : r.note}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
