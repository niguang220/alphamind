export function Label({ children, className = '' }) {
  return <div className={`label ${className}`}>{children}</div>
}

export function Rule({ className = '' }) {
  return <div className={`h-px bg-rule ${className}`} />
}

/** Small square status light. Squares, not circles — this is instrumentation, not chat. */
export function Dot({ color, live = false }) {
  return (
    <span
      className={`inline-block h-[7px] w-[7px] shrink-0 ${live ? 'node-live' : ''}`}
      style={{ background: color }}
    />
  )
}

export function Chip({ children, color = 'var(--color-ink-mute)', filled = false, title }) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1.5 border px-2 py-[3px] font-mono text-[10px] font-medium tracking-[0.1em] uppercase whitespace-nowrap"
      style={{
        color: filled ? 'var(--color-void)' : color,
        borderColor: filled ? color : `color-mix(in srgb, ${color} 42%, transparent)`,
        background: filled ? color : `color-mix(in srgb, ${color} 9%, transparent)`,
      }}
    >
      {children}
    </span>
  )
}

/** Horizontal meter used for confidence and fusion weights. */
export function Meter({ value, color = 'var(--color-teal)', muted = false, delay = 0 }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100
  if (muted) {
    return (
      <div
        className="h-[3px] w-full"
        style={{
          backgroundImage:
            'repeating-linear-gradient(90deg, var(--color-rule-bright) 0 4px, transparent 4px 9px)',
        }}
      />
    )
  }
  return (
    <div className="h-[3px] w-full bg-rule">
      <div
        className="sweep h-full"
        style={{ width: `${pct}%`, background: color, animationDelay: `${delay}ms` }}
      />
    </div>
  )
}

export function Field({ k, children, mono = true }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[7px]">
      <span className="label shrink-0">{k}</span>
      <span
        className={`min-w-0 text-right text-[12px] text-ink-soft ${mono ? 'tabular' : ''}`}
      >
        {children}
      </span>
    </div>
  )
}
