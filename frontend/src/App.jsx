import { useCallback, useEffect, useRef, useState } from 'react'
import { chat, health, isGuardrail } from './api'
import { SAMPLES } from './constants'
import TopBar from './components/TopBar'
import Composer from './components/Composer'
import Inspector from './components/Inspector'
import Pipeline from './components/Pipeline'
import GuardrailCard from './components/GuardrailCard'
import AnswerCard from './components/AnswerCard'

const USER_ID = 'desk-demo'

export default function App() {
  const [status, setStatus] = useState('checking')
  const [model, setModel] = useState(null)
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const convId = useRef(null)
  const streamEnd = useRef(null)

  useEffect(() => {
    let alive = true
    health()
      .then((h) => {
        if (!alive) return
        setStatus(h?.status === 'ok' ? 'online' : 'degraded')
        setModel(h?.model ?? null)
      })
      .catch(() => alive && setStatus('offline'))
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    streamEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, busy])

  const send = useCallback(
    async (message) => {
      setDraft('')
      setError(null)
      setBusy(true)
      const id = `t${Date.now()}`
      setTurns((prev) => [...prev, { id, kind: 'user', message }])

      try {
        const res = await chat({ message, userId: USER_ID, convId: convId.current })
        convId.current = res.conv_id
        const turn = { id: `${id}-a`, kind: 'answer', ...res }
        setTurns((prev) => [...prev, turn])
        setSelected(turn.id)
      } catch (e) {
        setError(e.message || 'request failed')
        setTurns((prev) => [...prev, { id: `${id}-e`, kind: 'error', message: e.message }])
      } finally {
        setBusy(false)
      }
    },
    [],
  )

  const answers = turns.filter((t) => t.kind === 'answer')
  const inspected = answers.find((t) => t.id === selected) ?? answers[answers.length - 1] ?? null
  const lastAnswer = answers[answers.length - 1] ?? null
  const pipelineState = busy ? 'running' : error && !lastAnswer ? 'error' : lastAnswer ? 'done' : 'idle'

  return (
    <div className="relative z-10 flex h-full flex-col">
      <TopBar status={status} model={model} />

      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto">
            <div className={`mx-auto flex w-full max-w-3xl flex-col px-6 ${turns.length === 0 ? 'min-h-full justify-center py-10' : 'py-7'}`}>
              {turns.length === 0 ? (
                <Welcome onPick={send} disabled={status === 'offline'} />
              ) : (
                <div className="space-y-5">
                  {turns.map((t) =>
                    t.kind === 'user' ? (
                      <UserTurn key={t.id} text={t.message} />
                    ) : t.kind === 'error' ? (
                      <ErrorTurn key={t.id} message={t.message} />
                    ) : isGuardrail(t) ? (
                      <div key={t.id} onClick={() => setSelected(t.id)}>
                        <GuardrailCard turn={t} />
                      </div>
                    ) : (
                      <AnswerCard
                        key={t.id}
                        turn={t}
                        selected={inspected?.id === t.id}
                        onSelect={() => setSelected(t.id)}
                      />
                    ),
                  )}
                </div>
              )}

              {(busy || lastAnswer) && (
                <div className="mt-5">
                  <Pipeline
                    state={pipelineState}
                    guardrail={!busy && lastAnswer ? isGuardrail(lastAnswer) : false}
                    latencyMs={!busy && lastAnswer ? lastAnswer.latency_ms : null}
                    agentLabel={!busy && lastAnswer ? lastAnswer.agent_type : null}
                  />
                </div>
              )}

              <div ref={streamEnd} />
            </div>
          </div>

          <Composer
            value={draft}
            onChange={setDraft}
            onSubmit={send}
            busy={busy}
            disabled={status === 'offline'}
          />
        </main>

        <div className="hidden w-[352px] shrink-0 xl:block">
          <Inspector turn={inspected} />
        </div>
      </div>
    </div>
  )
}

function UserTurn({ text }) {
  return (
    <div className="rise flex items-start gap-3">
      <span className="mt-[3px] shrink-0 font-mono text-[11px] tracking-[0.1em] text-ink-faint uppercase">
        You
      </span>
      <p className="min-w-0 border-l border-rule-bright pl-3.5 text-[14px] leading-[1.55] text-ink">
        {text}
      </p>
    </div>
  )
}

function ErrorTurn({ message }) {
  return (
    <div className="rise border border-ember/40 bg-ember/[0.05] px-5 py-3">
      <div className="label !text-ember">Request failed</div>
      <p className="tabular mt-1 text-[12px] text-ink-soft">{message}</p>
    </div>
  )
}

function Welcome({ onPick, disabled }) {
  return (
    <div className="rise">
      <p className="max-w-xl font-display text-[27px] leading-[1.28] text-ink">
        Securities research from three specialists — with{' '}
        <span className="text-teal italic">a clear view of how each answer was reached</span>.
      </p>
      <p className="mt-4 max-w-xl text-[13px] leading-[1.7] text-ink-mute">
        Ask about a company, a product, or the rules. Every answer shows which specialist
        took it and why. Ask it to pick a stock and it declines — by design.
      </p>

      <div className="mt-8 mb-3 flex items-center gap-3">
        <span className="label">Try one</span>
        <span className="h-px flex-1 bg-rule" />
      </div>

      <div className="grid gap-px bg-rule sm:grid-cols-2">
        {SAMPLES.map((s, i) => (
          <button
            key={s.text}
            disabled={disabled}
            onClick={() => onPick(s.text)}
            style={{ animationDelay: `${90 + i * 60}ms` }}
            className="rise group bg-desk px-4 py-3.5 text-left transition-colors hover:bg-panel disabled:opacity-40"
          >
            <span
              className="font-mono text-[9.5px] font-medium tracking-[0.16em] uppercase"
              style={{ color: s.color }}
            >
              {s.tag}
            </span>
            <p className="mt-1.5 text-[12.5px] leading-[1.5] text-ink-soft group-hover:text-ink">
              {s.text}
            </p>
            <p className="mt-1.5 text-[10px] text-ink-faint">{s.note}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
