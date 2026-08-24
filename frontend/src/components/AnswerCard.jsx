import { agentMeta } from '../constants'
import { Chip } from './Bits'
import Markdown from './Markdown'

export default function AnswerCard({ turn, selected, onSelect }) {
  const meta = agentMeta(turn.agent_type)
  const supporting = turn.supporting_agents ?? []

  return (
    <article
      onClick={onSelect}
      className={`rise cursor-pointer border bg-panel/70 transition-colors ${
        selected ? 'border-rule-bright' : 'border-rule hover:border-rule-bright/70'
      }`}
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-rule px-5 py-2.5">
        <span
          className="inline-flex items-center gap-2 font-mono text-[11px] font-semibold tracking-[0.16em] uppercase"
          style={{ color: meta.color }}
        >
          <span className="inline-block h-[7px] w-[7px]" style={{ background: meta.color }} />
          {meta.label}
        </span>

        <span className="text-[10px] text-ink-faint">/</span>
        <span className="font-mono text-[10px] tracking-[0.08em] text-ink-mute">
          {turn.intent}
        </span>

        {supporting.length > 0 && (
          <>
            <span className="text-[10px] text-ink-faint">+</span>
            {supporting.map((s) => (
              <Chip key={s} color={agentMeta(s).color}>
                {agentMeta(s).label} · supporting
              </Chip>
            ))}
          </>
        )}

        <span className="ml-auto flex items-center gap-2">
          {turn.knowledge_used && <Chip color="var(--color-teal)">kb hit</Chip>}
          {turn.escalated && <Chip color="var(--color-ember)">escalated</Chip>}
          <span className="tabular text-[10px] text-ink-faint">
            {Math.round(turn.latency_ms)} ms
          </span>
        </span>
      </header>

      <div className="px-5 py-4">
        <Markdown text={turn.response} />
      </div>
    </article>
  )
}
