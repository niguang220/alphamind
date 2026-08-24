import { agentMeta } from '../constants'
import { Chip } from './Bits'
import Markdown from './Markdown'

export default function AnswerCard({ turn, selected, onSelect }) {
  const meta = agentMeta(turn.agent_type)
  const supporting = turn.supporting_agents ?? []

  return (
    <article
      tabIndex={0}
      role="button"
      aria-pressed={selected}
      aria-label={`${meta.label} answer — show how it was decided`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onSelect?.()
        }
      }}
      onClick={onSelect}
      className="rise cursor-pointer bg-card shadow-card transition-colors"
      style={{ borderLeft: `3px solid ${selected ? meta.color : 'transparent'}` }}
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-hairline px-5 py-3">
        <span
          className="inline-flex items-center gap-2 font-mono text-meta font-semibold tracking-[0.16em] uppercase"
          style={{ color: meta.color }}
        >
          <span className="inline-block h-[7px] w-[7px]" style={{ background: meta.color }} />
          {meta.label}
        </span>

        <span className="text-meta text-ink-faint">/</span>
        <span className="font-mono text-meta tracking-[0.06em] text-ink-mute">
          {(turn.intent || '').replace(/_/g, ' ')}
        </span>

        {supporting.length > 0 && (
          <>
            <span className="text-meta text-ink-faint">+</span>
            {supporting.map((s) => (
              <Chip key={s} color={agentMeta(s).color}>
                {agentMeta(s).label} · supporting
              </Chip>
            ))}
          </>
        )}

        <span className="ml-auto flex items-center gap-2">
          {turn.knowledge_used && <Chip color="var(--color-teal)">cited</Chip>}
        </span>
      </header>

      <div className="px-5 py-5">
        <Markdown text={turn.response} />
      </div>
    </article>
  )
}
