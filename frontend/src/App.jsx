import { useCallback, useEffect, useRef, useState } from 'react'
import { chat, health, isGuardrail } from './api'
import { AGENTS, SAMPLES } from './constants'
import TopBar from './components/TopBar'
import Composer from './components/Composer'
import Inspector from './components/Inspector'
import Pipeline from './components/Pipeline'
import GuardrailCard from './components/GuardrailCard'
import AnswerCard from './components/AnswerCard'

const USER_ID = 'desk-demo'
const HEALTH_INTERVAL_MS = 20000

/**
 * One lane, shared by everything in the main column.
 *
 * The conversation, the composer and the welcome block all resolve to this width so they
 * share both edges. Before this the stream was a centred 720px measure inside a 1228px area
 * while the composer spanned the full width — six different left edges in one column.
 */
const LANE = 'mx-auto w-full max-w-[840px] px-8'

export default function App() {
  const [status, setStatus] = useState('checking')
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [selected, setSelected] = useState(null)
  const convId = useRef(null)
  const streamEnd = useRef(null)

  // Poll: a single check on mount left the header claiming ONLINE long after the backend died.
  useEffect(() => {
    let alive = true
    const check = () =>
      health()
        .then((h) => alive && setStatus(h?.status === 'ok' ? 'online' : 'degraded'))
        .catch(() => alive && setStatus('offline'))
    check()
    const id = setInterval(check, HEALTH_INTERVAL_MS)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    streamEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns, busy])

  const send = useCallback(async (message) => {
    setDraft('')
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
      setTurns((prev) => [...prev, { id: `${id}-e`, kind: 'error', error: describe(e) }])
    } finally {
      setBusy(false)
    }
  }, [])

  const answers = turns.filter((t) => t.kind === 'answer')
  const inspected = answers.find((t) => t.id === selected) ?? answers[answers.length - 1] ?? null

  return (
    <div className="relative z-10 flex h-full flex-col">
      <TopBar status={status} lane={LANE} />

      <div className="flex min-h-0 flex-1">
        <main className="grid min-w-0 flex-1 grid-rows-[1fr_auto]">
          <div className="stream min-h-0 overflow-y-auto">
            <div className={`${LANE} ${turns.length === 0 ? 'pt-[12vh] pb-10' : 'py-8'}`}>
              {turns.length === 0 ? (
                <Welcome onPick={send} disabled={status === 'offline'} />
              ) : (
                <div className="space-y-10">
                  {turns.map((t) =>
                    t.kind === 'user' ? (
                      <UserTurn key={t.id} text={t.message} />
                    ) : t.kind === 'error' ? (
                      <ErrorTurn key={t.id} error={t.error} />
                    ) : (
                      <Turn
                        key={t.id}
                        turn={t}
                        selected={inspected?.id === t.id}
                        onSelect={() => setSelected(t.id)}
                      />
                    ),
                  )}
                  {busy && <Pipeline state="running" />}
                </div>
              )}
              <div ref={streamEnd} />
            </div>
          </div>

          <div className="border-t border-rule bg-desk">
            <div className={LANE}>
              <Composer
                value={draft}
                onChange={setDraft}
                onSubmit={send}
                busy={busy}
                disabled={status === 'offline'}
              />
            </div>
          </div>
        </main>

        <div className="hidden w-[384px] shrink-0 xl:block">
          <Inspector turn={inspected} />
        </div>
      </div>
    </div>
  )
}

/** An answer and what it cost are one object, not two boxes spaced like unrelated turns. */
function Turn({ turn, selected, onSelect }) {
  const guardrail = isGuardrail(turn)
  return (
    <div onClick={onSelect} className="cursor-pointer">
      {guardrail ? <GuardrailCard turn={turn} /> : <AnswerCard turn={turn} selected={selected} />}
      <Pipeline
        state="done"
        guardrail={guardrail}
        timings={turn.timings}
        totalMs={turn.latency_ms}
      />
    </div>
  )
}

function describe(e) {
  const m = e?.message || ''
  if (/abort/i.test(m))
    return {
      title: 'That took too long',
      body: 'The desk did not answer in time. Your question is still in the box — press Enter to try again.',
    }
  if (/fetch|network|failed to fetch/i.test(m))
    return {
      title: "Can't reach the desk",
      body: 'The backend is not responding. Check that it is running on port 8000.',
    }
  return {
    title: "Couldn't complete that",
    body: `The desk returned an error (${m}). Try again in a moment.`,
  }
}

function UserTurn({ text }) {
  return (
    <div className="rise flex items-start gap-4">
      <span className="mt-[5px] shrink-0 font-mono text-[11px] tracking-[0.1em] text-ink-faint uppercase">
        You
      </span>
      <p className="min-w-0 text-[16px] leading-[1.5] text-ink">{text}</p>
    </div>
  )
}

function ErrorTurn({ error }) {
  return (
    <div className="rise border border-ember/40 bg-ember/[0.05] px-5 py-4">
      <div className="text-[13px] font-semibold text-ember">{error.title}</div>
      <p className="mt-1 text-[13px] leading-[1.6] text-ink-soft">{error.body}</p>
    </div>
  )
}

function Welcome({ onPick, disabled }) {
  return (
    <div className="rise">
      <p className="font-display text-[34px] leading-[1.22] text-ink">
        Securities research from four agents — and{' '}
        <span className="text-teal italic">a record of how each answer was reached</span>.
      </p>
      <p className="mt-5 max-w-[62ch] text-[14px] leading-[1.7] text-ink-mute">
        Ask about a company, a product, or the rules. Three specialists cover markets,
        research and compliance; a fourth takes anything that asks for a buy call and hands it
        to a licensed advisor instead.
      </p>

      <div className="mt-8 grid gap-x-8 gap-y-2.5 sm:grid-cols-2">
        {Object.entries(AGENTS).map(([key, a]) => (
          <div key={key} className="flex items-baseline gap-2.5">
            <span
              className="inline-block h-[7px] w-[7px] shrink-0"
              style={{ background: a.color }}
            />
            <span className="text-[12.5px] leading-[1.5]">
              <span style={{ color: a.color }}>{a.label}</span>
              <span className="text-ink-faint"> — {a.blurb}</span>
            </span>
          </div>
        ))}
      </div>

      <div className="mt-10 mb-3 flex items-center gap-3">
        <span className="label">Start here</span>
        <span className="h-px flex-1 bg-rule" />
      </div>

      <div className="grid gap-px bg-rule sm:grid-cols-2">
        {SAMPLES.map((s, i) => (
          <button
            key={s.text}
            disabled={disabled}
            onClick={() => onPick(s.text)}
            style={{ animationDelay: `${90 + i * 60}ms` }}
            /* flex-col defeats the UA's vertical centring, which dropped a short tile 10px
               below a wrapping neighbour in the same row */
            className="rise group flex flex-col items-start bg-desk px-5 py-4 text-left transition-colors hover:bg-panel disabled:opacity-40"
          >
            <span
              className="font-mono text-[10px] font-medium tracking-[0.16em] uppercase"
              style={{ color: s.color }}
            >
              {s.tag}
            </span>
            <p className="mt-2 text-[13px] leading-[1.5] text-ink-soft group-hover:text-ink">
              {s.text}
            </p>
            <p className="mt-2 text-[11px] text-ink-faint">{s.note}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
