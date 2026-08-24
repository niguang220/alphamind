import { STAGES } from '../constants'

/**
 * What the request actually spent, stage by stage.
 *
 * Every number here is measured on the server and returned in `timings`; the bar widths are
 * those milliseconds to scale. On a guardrail interception the agent stage costs ~0ms and the
 * strip shows it plainly — the refusal is a fixed template, so nothing was generated. It also
 * shows the part that used to be hidden: classification and retrieval run before any agent
 * does, and on a short turn they are most of the wall clock.
 */
export default function Pipeline({ state, guardrail, timings, totalMs }) {
  if (state === 'running') return <Running />
  if (state === 'error') return <Failed />
  if (!timings) return null

  const total = totalMs || timings.total || 1
  const accent = guardrail ? 'var(--color-brass)' : 'var(--color-teal)'

  return (
    <div className="border-t border-rule bg-void/40 px-5 py-3.5">
      <div className="flex items-baseline justify-between gap-4">
        <div className="label">Where the time went</div>
        <div className="tabular text-[11px] text-ink-soft">
          <span style={{ color: accent }}>{fmt(total)}</span> total
        </div>
      </div>

      <div className="mt-3 flex h-[5px] w-full">
        {STAGES.map((s) => {
          const ms = timings[s.key] ?? 0
          const pct = (ms / total) * 100
          if (pct < 0.4) return null
          return (
            <div
              key={s.id}
              className="sweep h-full"
              style={{
                width: `${pct}%`,
                background: s.key === 'orchestration' ? accent : 'var(--color-teal-dim)',
                marginRight: '2px',
              }}
              title={`${s.label} ${fmt(ms)}`}
            />
          )
        })}
      </div>

      <div className="mt-2.5 grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-4">
        {STAGES.map((s) => {
          const ms = timings[s.key] ?? 0
          const zero = ms < 1
          return (
            <div key={s.id} className="flex items-baseline justify-between gap-2">
              <span
                className={`font-mono text-[10px] tracking-[0.1em] uppercase ${
                  zero ? 'text-ink-faint' : 'text-ink-mute'
                }`}
              >
                {s.label}
              </span>
              <span
                className="tabular text-[11px]"
                style={{ color: zero ? 'var(--color-ink-faint)' : 'var(--color-ink-soft)' }}
              >
                {fmt(ms)}
              </span>
            </div>
          )
        })}
      </div>

      {guardrail && (
        <p className="mt-3 border-t border-brass/25 pt-2.5 text-[12px] leading-[1.55] text-brass/90">
          The agent stage cost nothing — the request was screened out before any agent ran, and
          the refusal is a fixed compliance template rather than generated text.
        </p>
      )}
    </div>
  )
}

const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`)

function Running() {
  return (
    <div className="rise flex flex-wrap items-center gap-x-3 gap-y-1 border border-rule bg-panel px-5 py-3.5">
      <span className="node-live inline-block h-[7px] w-[7px] shrink-0 bg-teal" />
      <span className="text-[13px] text-ink-mute">
        Working<span className="blink">…</span>
      </span>
      <span className="text-[11px] text-ink-faint">
        classifying, screening, then routing to an agent
      </span>
    </div>
  )
}

function Failed() {
  return (
    <div className="rise border border-ember/40 bg-ember/[0.05] px-5 py-3">
      <div className="label !text-ember">Request failed</div>
    </div>
  )
}
