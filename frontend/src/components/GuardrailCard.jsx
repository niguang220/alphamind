import { Chip } from './Bits'

/**
 * Splits the fixed compliance template the orchestrator returns into its parts:
 * a lead refusal, the numbered risk disclosure, and the closing offer.
 * Falls back to plain paragraphs if the template ever changes.
 */
function parseTemplate(text) {
  const lines = (text || '').split('\n').map((l) => l.trim()).filter(Boolean)
  const lead = []
  const items = []
  const tail = []
  for (const line of lines) {
    const m = line.match(/^(\d+)\.\s*(.*)$/)
    if (m) items.push({ n: m[1], body: m[2].replace(/[;.]$/, '') })
    else if (items.length) tail.push(line)
    else lead.push(line)
  }
  return { lead, items, tail }
}

export default function GuardrailCard({ turn }) {
  const { lead, items, tail } = parseTemplate(turn.response)

  return (
    <article className="rise relative overflow-hidden border border-brass/45 bg-[#12100a]">
      {/* corner brackets — regulatory stamp feel without literal caution tape */}
      <span className="pointer-events-none absolute top-0 left-0 h-3 w-3 border-t border-l border-brass/70" />
      <span className="pointer-events-none absolute top-0 right-0 h-3 w-3 border-t border-r border-brass/70" />
      <span className="pointer-events-none absolute bottom-0 left-0 h-3 w-3 border-b border-l border-brass/70" />
      <span className="pointer-events-none absolute right-0 bottom-0 h-3 w-3 border-r border-b border-brass/70" />

      <header className="hazard flex flex-wrap items-center justify-between gap-3 border-b border-brass/35 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <ShieldGlyph />
          <h3 className="font-mono text-[11px] font-semibold tracking-[0.2em] text-brass uppercase">
            Compliance Guardrail
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <Chip color="var(--color-brass)">intent · advice_request</Chip>
          <Chip color="var(--color-brass)" filled>
            escalated
          </Chip>
        </div>
      </header>

      <div className="px-5 py-5">
        {lead.map((p, i) => (
          <p
            key={i}
            className={`text-[14px] leading-[1.65] ${i === 0 ? 'text-ink' : 'mt-2.5 text-ink-soft'}`}
          >
            {i === 0 ? <Emphasize text={p} /> : p}
          </p>
        ))}

        {items.length > 0 && (
          <ol className="mt-5 space-y-0">
            {items.map((it, i) => (
              <li
                key={it.n}
                className="rise flex gap-3.5 border-l border-brass/30 py-2.5 pl-4"
                style={{ animationDelay: `${120 + i * 70}ms` }}
              >
                <span className="tabular mt-[3px] shrink-0 text-[11px] text-brass/70">
                  {String(it.n).padStart(2, '0')}
                </span>
                <span className="text-[13px] leading-[1.6] text-ink-soft">{it.body}</span>
              </li>
            ))}
          </ol>
        )}

        {tail.length > 0 && (
          <div className="mt-5 border-t border-rule pt-4">
            <div className="label mb-2">What I can do instead</div>
            {tail.map((p, i) => (
              <p key={i} className="text-[13px] leading-[1.6] text-ink-mute">
                {p}
              </p>
            ))}
          </div>
        )}
      </div>

      <footer className="border-t border-brass/25 bg-brass/[0.04] px-5 py-3">
        <span className="text-[12.5px] text-brass/90">
          Escalated to a licensed advisor — intercepted before any agent ran.
        </span>
      </footer>
    </article>
  )
}

/** Bolds the compliance sentence that the evaluator also keys on. */
function Emphasize({ text }) {
  const key = 'does not constitute investment advice'
  const idx = text.toLowerCase().indexOf(key)
  if (idx === -1) return text
  return (
    <>
      {text.slice(0, idx)}
      <strong className="font-semibold text-brass">{text.slice(idx, idx + key.length)}</strong>
      {text.slice(idx + key.length)}
    </>
  )
}

function ShieldGlyph() {
  return (
    <svg width="15" height="17" viewBox="0 0 15 17" fill="none" aria-hidden>
      <path
        d="M7.5 1L14 3.4v5.1c0 4-2.8 6.6-6.5 7.9C3.8 15.1 1 12.5 1 8.5V3.4L7.5 1Z"
        stroke="var(--color-brass)"
        strokeWidth="1.1"
        strokeLinejoin="round"
      />
      <path d="M4.6 8.4l2 2 3.8-4" stroke="var(--color-brass)" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
