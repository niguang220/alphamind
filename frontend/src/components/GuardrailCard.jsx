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
    <article
      className="rise relative overflow-hidden bg-[#12100a] shadow-card"
      style={{ borderLeft: '3px solid var(--color-brass)' }}
    >
      {/* One hazard signal, as a 4px band at the top edge. Running the stripes behind the
          header meant they passed through a transparent chip and behind its text. */}
      <span className="hazard block h-[4px] w-full" />

      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-brass/30 px-5 py-3">
        <h3 className="font-mono text-meta font-semibold tracking-[0.2em] text-brass uppercase">
          Compliance Guardrail
        </h3>
        <Chip color="var(--color-brass)" filled>
          Advice request · intercepted
        </Chip>
      </header>

      <div className="px-5 py-5">
        {lead.map((p, i) => (
          <p
            key={i}
            className={`text-body leading-[1.65] ${i === 0 ? 'text-ink' : 'mt-2.5 text-ink-soft'}`}
          >
            {i === 0 ? <Emphasize text={p} /> : p}
          </p>
        ))}

        {items.length > 0 && (
          <ol className="mt-5 space-y-0">
            {items.map((it, i) => (
              <li
                key={it.n}
                className="rise flex gap-3.5 border-l border-brass/25 py-2.5 pl-4"
                style={{ animationDelay: `${120 + i * 70}ms` }}
              >
                <span className="tabular mt-[3px] shrink-0 text-meta text-brass/70">
                  {String(it.n).padStart(2, '0')}
                </span>
                <span className="text-ui leading-[1.6] text-ink-soft">{it.body}</span>
              </li>
            ))}
          </ol>
        )}

        {tail.length > 0 && (
          <div className="mt-5 border-t border-rule pt-4">
            <div className="label mb-2">What I can do instead</div>
            {tail.map((p, i) => (
              <p key={i} className="text-ui leading-[1.6] text-ink-mute">
                {p}
              </p>
            ))}
          </div>
        )}
      </div>

      <footer className="border-t border-brass/25 bg-brass/[0.04] px-5 py-3">
        <span className="text-ui text-brass/90">
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

