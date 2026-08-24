import { useEffect, useState } from 'react'
import { STAGES } from '../constants'

/**
 * Orchestrator pipeline stepper.
 *
 * The four stages mirror AgentOrchestrator.run(). While a request is in flight the
 * stepper advances on indicative pacing; when the response lands it snaps to the
 * truth the backend reported — and a guardrail interception visibly stops the run
 * at stage 2 with generation struck out, which is the whole point of the guardrail.
 */
export default function Pipeline({ state, guardrail, latencyMs, agentLabel }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (state !== 'running') return
    const t0 = performance.now()
    const id = setInterval(() => setElapsed(performance.now() - t0), 60)
    return () => clearInterval(id)
  }, [state])

  const done = state === 'done'
  const failed = state === 'error'

  const statusOf = (idx) => {
    if (failed) return idx === 0 ? 'error' : 'idle'
    if (done) {
      if (guardrail) {
        if (idx === 0) return 'passed'
        if (idx === 1) return 'intercepted'
        return 'skipped'
      }
      return 'passed'
    }
    // running
    const reached = STAGES.filter((s) => elapsed >= s.at).length - 1
    if (idx < reached) return 'passed'
    if (idx === reached) return 'active'
    return 'idle'
  }

  const accent = guardrail && done ? 'var(--color-brass)' : 'var(--color-teal)'

  return (
    <div className="border border-rule bg-panel/60 px-4 py-3">
      <div className="flex items-center justify-between gap-4">
        <Label>How this was handled</Label>
        <span className="tabular text-[10px] text-ink-faint">
          {done && latencyMs != null ? (
            <>
              <span style={{ color: accent }}>{Math.round(latencyMs)}</span> ms end-to-end
            </>
          ) : state === 'running' ? (
            <span className="text-ink-mute">
              {(elapsed / 1000).toFixed(1)}s<span className="blink"> ▍</span>
            </span>
          ) : failed ? (
            <span style={{ color: 'var(--color-ember)' }}>failed</span>
          ) : null}
        </span>
      </div>

      <div className="mt-3 flex items-stretch">
        {STAGES.map((stage, i) => {
          const st = statusOf(i)
          const color =
            st === 'intercepted'
              ? 'var(--color-brass)'
              : st === 'error'
                ? 'var(--color-ember)'
                : st === 'passed' || st === 'active'
                  ? 'var(--color-teal)'
                  : 'var(--color-rule-bright)'
          const dim = st === 'skipped' || st === 'idle'
          return (
            <div key={stage.id} className="flex min-w-0 flex-1 items-start">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-block h-[7px] w-[7px] shrink-0 ${st === 'active' ? 'node-live' : ''}`}
                    style={{
                      background: dim ? 'transparent' : color,
                      border: dim ? '1px solid var(--color-rule-bright)' : 'none',
                    }}
                  />
                  <span
                    className={`truncate font-mono text-[10px] font-medium tracking-[0.12em] uppercase ${
                      st === 'skipped' ? 'text-ink-faint line-through' : dim ? 'text-ink-faint' : ''
                    }`}
                    style={{ color: dim ? undefined : color }}
                  >
                    {stage.label}
                  </span>
                </div>
                <div
                  className={`mt-1.5 truncate pl-[15px] text-[10px] ${
                    st === 'skipped' ? 'text-ink-faint/70 line-through' : 'text-ink-faint'
                  }`}
                >
                  {st === 'intercepted'
                    ? 'intercepted'
                    : st === 'skipped'
                      ? 'skipped'
                      : st === 'passed' && i === 3 && agentLabel
                        ? agentLabel.toLowerCase()
                        : stage.detail}
                </div>
              </div>
              {i < STAGES.length - 1 && (
                <div
                  className="mt-[3px] h-px w-4 shrink-0 sm:w-6"
                  style={{
                    background:
                      statusOf(i + 1) === 'skipped' || statusOf(i + 1) === 'idle'
                        ? 'var(--color-rule)'
                        : color,
                  }}
                />
              )}
            </div>
          )
        })}
      </div>

      {done && guardrail && (
        <div className="mt-3 border-t border-brass/25 pt-2.5 text-[11px] text-brass/85">
          Stopped right here — no model was ever asked to answer this.
        </div>
      )}
    </div>
  )
}

function Label({ children }) {
  return <div className="label">{children}</div>
}
