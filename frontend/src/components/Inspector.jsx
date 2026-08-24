import { agentMeta, EVAL_SNAPSHOT } from '../constants'
import { isGuardrail } from '../api'
import { Chip, Meter, Rule } from './Bits'
import FusionBars from './FusionBars'

export default function Inspector({ turn }) {
  return (
    <aside className="flex h-full min-h-0 w-full flex-col border-l border-rule bg-desk">
      <div className="flex items-center justify-between border-b border-rule px-6 py-3">
        <div className="label">How this was decided</div>
        <div className="text-[11px] text-ink-faint">{turn ? 'this answer' : 'nothing yet'}</div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
        {turn ? <TurnRecord turn={turn} /> : <IdleState />}
        <Rule className="my-7" />
        <EvalSnapshot />
      </div>
    </aside>
  )
}

function TurnRecord({ turn }) {
  const guardrail = isGuardrail(turn)
  const primary = agentMeta(turn.primary_agent || turn.agent_type)
  const supporting = turn.supporting_agents ?? []
  const entities = Object.entries(turn.entities ?? {}).filter(([, v]) => v?.length)
  const scores = parseScores(turn.routing_reason)

  return (
    <div className="space-y-7">
      {/* who answered — the anchor the panel was missing */}
      <section>
        <div className="label mb-2.5">Answered by</div>
        <div className="flex items-baseline gap-3">
          <span className="font-display text-[26px] leading-none" style={{ color: primary.color }}>
            {primary.label}
          </span>
          {/* The score is a sum of bonuses, not a probability — it belongs next to the
              scores it beat, below, rather than alone up here looking like a percentage. */}
          <span className="tabular text-[12px] text-ink-faint">
            {guardrail ? 'rule' : turn.routing_score != null ? turn.routing_score.toFixed(2) : '—'}
          </span>
        </div>
        <p className="mt-1.5 text-[12px] text-ink-mute">{primary.blurb}</p>
        {supporting.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {supporting.map((s) => (
              <Chip key={s} color={agentMeta(s).color}>
                {agentMeta(s).label} · supporting
              </Chip>
            ))}
          </div>
        )}
      </section>

      {/* why — scores parsed out of the routing reason instead of dumped as a log line */}
      {scores.length > 0 && (
        <section>
          <div className="label mb-2.5">Why</div>
          <div className="space-y-2">
            {scores.map(([name, val], i) => {
              const meta = agentMeta(name)
              const top = i === 0
              return (
                <div key={name} className="flex items-center gap-3">
                  <span
                    className={`w-20 shrink-0 font-mono text-[11px] tracking-[0.06em] uppercase ${
                      top ? 'text-ink-soft' : 'text-ink-faint'
                    }`}
                  >
                    {meta.label}
                  </span>
                  <div className="flex-1">
                    <Meter
                      value={val / 1.5}
                      color={top ? meta.color : 'var(--color-rule-bright)'}
                      delay={i * 80}
                    />
                  </div>
                  <span
                    className="tabular w-10 shrink-0 text-right text-[11px]"
                    style={{ color: top ? meta.color : 'var(--color-ink-faint)' }}
                  >
                    {val.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>
          <p className="mt-2 text-[11px] leading-[1.5] text-ink-faint">
            Additive scores from intent, keywords and entities — unbounded, and not
            probabilities. Highest takes the turn; a close second is brought in to support.
          </p>
        </section>
      )}

      {guardrail && (
        <section>
          <div className="label mb-2">Why</div>
          <p className="border-l-2 border-brass/60 py-1 pl-3 text-[12.5px] leading-[1.6] text-brass/90">
            Classified as an advice request and screened out before routing. No agent ran.
          </p>
        </section>
      )}

      <Rule />

      <section>
        <div className="label mb-2.5">What was asked</div>
        <dl className="space-y-2">
          <Row k="Intent">{pretty(turn.intent)}</Row>
          <Row k="Area">{pretty(turn.intent_group)}</Row>
          <Row k="Sources">
            {turn.knowledge_used ? (
              <span className="text-teal">
                Knowledge base cited
                {turn.knowledge_score != null && (
                  <span className="text-ink-faint"> · {turn.knowledge_score.toFixed(2)}</span>
                )}
              </span>
            ) : (
              <span className="text-ink-mute">Model knowledge only</span>
            )}
          </Row>
          <Row k="Handoff">
            {turn.escalated ? (
              <span className="text-ink-soft">
                {guardrail ? 'Escalated — advice request' : 'Offered a licensed advisor'}
              </span>
            ) : (
              <span className="text-ink-mute">None</span>
            )}
          </Row>
        </dl>
      </section>

      {entities.length > 0 && (
        <>
          <Rule />
          <section>
            <div className="label mb-2.5">Detected</div>
            <div className="space-y-2.5">
              {entities.map(([k, v]) => (
                <div key={k}>
                  <div className="label !text-[10px]">{k.replace(/_/g, ' ')}</div>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {v.map((x) => (
                      <span
                        key={x}
                        className="tabular border border-rule bg-raised px-2 py-[3px] text-[11px] text-ink-soft"
                      >
                        {x}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      <Rule />

      <FusionBars scores={turn.intent_source_scores} confidence={turn.intent_confidence} />
    </div>
  )
}

/** `intent=x, ..., scores=[research=1.40, market=0.22]` -> sorted [[name, value]] */
function parseScores(reason) {
  const m = /scores=\[([^\]]+)\]/.exec(reason || '')
  if (!m) return []
  return m[1]
    .split(',')
    .map((p) => p.trim().split('='))
    .filter((p) => p.length === 2 && p[1] !== '' && !Number.isNaN(Number(p[1])))
    .map(([k, v]) => [k, Number(v)])
    .sort((a, b) => b[1] - a[1])
}

const pretty = (s) => (s || '').replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())

function Row({ k, children }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="label shrink-0 !text-[11px]">{k}</dt>
      <dd className="min-w-0 text-right text-[12.5px] text-ink-soft">{children}</dd>
    </div>
  )
}

function IdleState() {
  return (
    <p className="text-[13px] leading-[1.7] text-ink-mute">
      Every answer leaves a record here — the agent that took it, the scores it beat, and what
      it drew on.
    </p>
  )
}

function EvalSnapshot() {
  const { headline, quality, ranAt, model, caveat } = EVAL_SNAPSHOT
  return (
    <section>
      <div className="flex items-baseline justify-between">
        <div className="label">Benchmark</div>
        <div className="tabular text-[11px] text-ink-faint">{ranAt}</div>
      </div>

      <div className="mt-3 space-y-2">
        {headline.map((m) => (
          <div key={m.key} className="flex items-baseline justify-between gap-3">
            <span className="text-[12px] text-ink-mute">{m.key}</span>
            <span className="tabular text-[13px] text-ink-soft">
              {m.pass}
              <span className="text-ink-faint">/{m.of}</span>
            </span>
          </div>
        ))}
      </div>

      <div className="mt-4 space-y-1.5">
        {quality.map((m) => (
          <div key={m.key} className="flex items-center gap-3">
            <span className="label w-24 shrink-0 !text-[10px]">{m.key}</span>
            <div className="flex-1">
              {/* rescaled to 0.80–1.00; on a 0–1 track these four are indistinguishable */}
              <Meter value={(m.value - 0.8) / 0.2} color="var(--color-teal-dim)" />
            </div>
            <span className="tabular w-9 shrink-0 text-right text-[11px] text-ink-mute">
              {m.value.toFixed(2)}
            </span>
          </div>
        ))}
      </div>

      <p className="mt-3 text-[11px] leading-[1.55] text-ink-faint">
        {caveat} Quality scale 0.80–1.00, judged on {model}.
      </p>
    </section>
  )
}
