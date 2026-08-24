import { Dot } from './Bits'

export default function TopBar({ status, model }) {
  const online = status === 'online'
  const color = online
    ? 'var(--color-teal)'
    : status === 'checking'
      ? 'var(--color-brass)'
      : 'var(--color-ember)'

  return (
    <header className="flex items-center justify-between gap-6 border-b border-rule bg-desk/90 px-6 py-3 backdrop-blur">
      <div className="flex items-baseline gap-4">
        <h1 className="font-display text-[25px] leading-none tracking-[-0.01em] text-ink">
          Alpha<span className="text-teal">Mind</span>
        </h1>
        <span className="hidden font-mono text-[10px] tracking-[0.14em] text-ink-faint uppercase sm:block">
          Securities Research Desk
        </span>
      </div>

      <div className="flex items-center gap-5">
        <p className="hidden max-w-md text-right text-[10.5px] leading-[1.5] text-ink-faint lg:block">
          Research information and investor education. Not investment advice — no stock
          recommendations, no return guarantees, no trading on your behalf.
        </p>
        <div className="flex shrink-0 items-center gap-2 border border-rule bg-panel px-2.5 py-1.5">
          <Dot color={color} live={online} />
          <span className="font-mono text-[10px] tracking-[0.1em] text-ink-mute uppercase">
            {status}
          </span>
          {model && (
            <>
              <span className="text-ink-faint/50">·</span>
              <span className="tabular text-[10px] text-ink-faint">{model}</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
