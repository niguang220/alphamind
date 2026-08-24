import { Dot } from './Bits'

export default function TopBar({ status, lane }) {
  const online = status === 'online'
  const color = online
    ? 'var(--color-teal)'
    : status === 'checking'
      ? 'var(--color-brass)'
      : 'var(--color-ember)'
  const word = { online: 'Live', checking: 'Connecting', degraded: 'Limited', offline: 'Offline' }[status]

  return (
    <header className="border-b border-rule bg-desk">
      <div className="flex items-center justify-between gap-6 py-3 pr-6 pl-8">
        <div className="flex items-baseline gap-4">
          <h1 className="font-display text-[25px] leading-none tracking-[-0.01em] text-ink">
            Alpha<span className="text-teal">Mind</span>
          </h1>
          <span className="hidden font-mono text-[10px] tracking-[0.14em] text-ink-faint uppercase sm:block">
            Research Desk · China A-Shares
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Dot color={color} live={online} />
          <span className="font-mono text-[11px] tracking-[0.08em] text-ink-mute">{word}</span>
        </div>
      </div>
      {/* The disclosure sits in the main lane rather than straddling the inspector's edge,
          and reads at body weight because it is the sentence a compliance reader looks for. */}
      <div className={`${lane} pb-3`}>
        <p className="text-[12px] leading-[1.5] text-ink-mute">
          Research and investor education. Not investment advice — no stock recommendations,
          no return guarantees, no trading on your behalf.
        </p>
      </div>
    </header>
  )
}
