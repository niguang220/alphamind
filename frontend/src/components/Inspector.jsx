import { agentMeta, EVAL_SNAPSHOT } from '../constants'
import { isGuardrail } from '../api'
import { Chip, Field, Meter, Rule } from './Bits'
import FusionBars from './FusionBars'

export default function Inspector({ turn }) {
  return (
    <aside className="flex h-full min-h-0 w-full flex-col border-l border-rule bg-desk">
      <div className="flex items-center justify-between border-b border-rule px-5 py-3">
        <div className="label">Inspector</div>
        <div className="text-[10px] text-ink-faint">
          {turn ? 'latest turn' : 'idle'}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {turn ? <TurnTelemetry turn={turn} /> : <IdleState />}
        <Rule className="my-6" />
        <EvalSnapshot />
      </div>
    </aside>
  )
}

function TurnTelemetry({ turn }) {
  const guardrail = isGuardrail(turn)
  const primary = agentMeta(turn.primary_agent || turn.agent_type)
  const supporting = turn.supporting_agents ?? []
  const entities = Object.entries(turn.entities ?? {}).filter(([, v]) => v?.length)

  return (
    <div className="space-y-6">
      {/* routing */}
      <section>
        <div className="label mb-3">Routing decision</div>

        <div className="flex flex-wrap items-center gap-2">
          <Chip color={primary.color} filled>
            {primary.label} · primary
          </Chip>
          {supporting.length ? (
            supporting.map((s) => (
              <Chip key={s} color={agentMeta(s).color}>
                {agentMeta(s).label} · supporting
              </Chip>
            ))
          ) : (
            <span className="text-[11px] text-ink-faint">no supporting agents</span>
          )}
        </div>

        <div className="mt-4">
          <div className="flex items-baseline justify-between">
            <span className="label">Routing confidence</span>
            <span className="tabular text-[11px]" style={{ color: primary.color }}>
              {(turn.routing_confidence ?? 0).toFixed(2)}
            </span>
          </div>
          <div className="mt-1.5">
            <Meter value={turn.routing_confidence} color={primary.color} />
          </div>
        </div>

        <div className="mt-4">
          <div className="label mb-1.5">Routing reason</div>
          <p
            className={`tabular border-l-2 py-1 pl-3 text-[10.5px] leading-[1.65] break-words ${
              guardrail ? 'border-brass/60 text-brass/90' : 'border-rule-bright text-ink-mute'
            }`}
          >
            {turn.routing_reason || '—'}
          </p>
        </div>
      </section>

      <Rule />

      {/* classification */}
      <section>
        <div className="label mb-1">Classification</div>
        <div className="divide-y divide-rule/60">
          <Field k="intent">{turn.intent}</Field>
          <Field k="group">{turn.intent_group}</Field>
          <Field k="agents">{(turn.agent_types ?? []).join(' · ') || '—'}</Field>
          <Field k="escalated">
            <span style={{ color: turn.escalated ? 'var(--color-ember)' : 'var(--color-ink-mute)' }}>
              {String(turn.escalated)}
            </span>
          </Field>
          <Field k="knowledge">
            <span style={{ color: turn.knowledge_used ? 'var(--color-teal)' : 'var(--color-ink-mute)' }}>
              {turn.knowledge_used ? 'retrieved' : 'not used'}
            </span>
          </Field>
          <Field k="latency">{Math.round(turn.latency_ms)} ms</Field>
        </div>
        {turn.escalated && !guardrail && (
          <p className="mt-2 text-[10px] leading-[1.5] text-ink-faint">
            Flagged because the answer itself offers a human advisor — not the guardrail.
          </p>
        )}
      </section>

      <Rule />

      <FusionBars scores={turn.intent_source_scores} confidence={turn.intent_confidence} />

      {entities.length > 0 && (
        <>
          <Rule />
          <section>
            <div className="label mb-2.5">Extracted entities</div>
            <div className="space-y-2">
              {entities.map(([k, v]) => (
                <div key={k} className="flex items-start gap-2.5">
                  <span className="label mt-[3px] w-16 shrink-0">{k}</span>
                  <div className="flex flex-wrap gap-1.5">
                    {v.map((x) => (
                      <span
                        key={x}
                        className="tabular border border-rule bg-raised px-1.5 py-[2px] text-[10px] text-ink-soft"
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
    </div>
  )
}

function IdleState() {
  return (
    <div className="py-2">
      <p className="text-[12px] leading-[1.7] text-ink-mute">
        Send a message and this panel opens up the answer: who handled it, how sure the
        system was, and what it leaned on.
      </p>
      <ul className="mt-4 space-y-2 text-[11px] text-ink-faint">
        {[
          'Which agent answered, and how it was classified',
          'How confident the routing was',
          'Whether it drew on the knowledge base',
          'Whether it was handed to a human',
        ].map((t) => (
          <li key={t} className="flex gap-2.5">
            <span className="mt-[7px] inline-block h-px w-2.5 shrink-0 bg-rule-bright" />
            {t}
          </li>
        ))}
      </ul>
    </div>
  )
}

function EvalSnapshot() {
  const { headline, quality, ranAt, model, cases } = EVAL_SNAPSHOT
  return (
    <section>
      <div className="flex items-baseline justify-between">
        <div className="label">Latest evaluation</div>
        <div className="tabular text-[10px] text-ink-faint">{ranAt}</div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-px bg-rule">
        {headline.map((m) => (
          <div key={m.key} className="bg-desk px-2 py-2.5 text-center">
            <div className="tabular text-[17px] leading-none text-teal">
              {m.value.toFixed(2)}
            </div>
            <div className="label mt-1.5 !text-[8.5px] !tracking-[0.08em]">
              {m.key.replace(/_/g, ' ')}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 space-y-1.5">
        {quality.map((m) => (
          <div key={m.key} className="flex items-center gap-3">
            <span className="label w-20 shrink-0 !text-[9px]">{m.key}</span>
            <div className="flex-1">
              <Meter value={m.value} color="var(--color-teal-dim)" />
            </div>
            <span className="tabular w-9 shrink-0 text-right text-[10px] text-ink-mute">
              {m.value.toFixed(2)}
            </span>
          </div>
        ))}
      </div>

      <p className="mt-3 text-[10px] leading-[1.55] text-ink-faint">
        Last full run: {cases}, scored by an LLM judge on {model}.
      </p>
    </section>
  )
}
