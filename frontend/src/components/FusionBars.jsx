import { Meter } from './Bits'

/**
 * Intent recognition is a weighted vote over three routes. On an Anthropic-compatible
 * third-party provider the embedding route is switched off at construction
 * (IntentRecognizer.__init__) and its weight is redistributed, so its score is
 * always 0. Showing that explicitly is the honest read — hiding the row would make
 * a deliberate degradation look like a broken bar.
 */
const ROUTES = [
  { key: 'llm', label: 'LLM', weight: '0.85', note: 'few-shot semantic classifier' },
  { key: 'embedding', label: 'Embedding', weight: '—', note: null },
  { key: 'pattern', label: 'Pattern', weight: '0.15', note: 'keyword fallback, synchronous' },
]

export default function FusionBars({ scores, confidence }) {
  const s = scores ?? {}
  const embeddingOff = !s.embedding

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <div className="label">Intent fusion</div>
        <div className="tabular text-[11px] text-ink-soft">
          conf <span className="text-teal">{(confidence ?? 0).toFixed(2)}</span>
        </div>
      </div>

      <div className="mt-3 space-y-3">
        {ROUTES.map((r, i) => {
          const off = r.key === 'embedding' && embeddingOff
          const val = s[r.key] ?? 0
          return (
            <div key={r.key}>
              <div className="flex items-baseline justify-between gap-3">
                <span
                  className={`font-mono text-[10px] tracking-[0.1em] uppercase ${
                    off ? 'text-ink-faint' : 'text-ink-soft'
                  }`}
                >
                  {r.label}
                </span>
                <span className="tabular text-[10px] text-ink-faint">
                  {off ? 'disabled' : val.toFixed(2)}
                  <span className="ml-2 text-ink-faint/60">w {r.weight}</span>
                </span>
              </div>
              <div className="mt-1.5">
                <Meter
                  value={val}
                  muted={off}
                  color={r.key === 'llm' ? 'var(--color-teal)' : 'var(--color-azure)'}
                  delay={i * 90}
                />
              </div>
              {off && (
                <p className="mt-1.5 text-[10px] leading-[1.5] text-ink-faint">
                  Not available on this provider. Its weight goes to the other two.
                </p>
              )}
              {!off && r.note && (
                <p className="mt-1.5 text-[10px] text-ink-faint/80">{r.note}</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
