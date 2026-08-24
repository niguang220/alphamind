import { useEffect, useRef } from 'react'

export default function Composer({ value, onChange, onSubmit, busy, disabled }) {
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`
  }, [value])

  const submit = () => {
    if (busy || disabled || !value.trim()) return
    onSubmit(value.trim())
  }

  return (
    <div className="border-t border-rule bg-desk px-6 py-4">
      <div
        className={`flex items-end gap-3 border bg-panel px-4 py-3 transition-colors ${
          busy ? 'border-teal/35' : 'border-rule focus-within:border-rule-bright'
        }`}
      >
        <span className="pb-[3px] font-mono text-[13px] leading-none text-teal/70 select-none">
          ›
        </span>
        <textarea
          ref={ref}
          rows={1}
          value={value}
          disabled={busy || disabled}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          placeholder={
            disabled ? 'Backend unreachable' : 'Ask about a security, a product, or the rules…'
          }
          className="max-h-42 min-h-[20px] flex-1 resize-none bg-transparent text-[13.5px] leading-[1.5] text-ink placeholder:text-ink-faint focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={submit}
          disabled={busy || disabled || !value.trim()}
          className="shrink-0 border border-rule-bright px-3 py-[5px] font-mono text-[10px] font-medium tracking-[0.14em] text-ink-mute uppercase transition-colors enabled:hover:border-teal enabled:hover:text-teal disabled:opacity-35"
        >
          {busy ? 'running' : 'send'}
        </button>
      </div>
      <div className="mt-2 flex items-center justify-between px-1">
        <span className="text-[10px] text-ink-faint">
          Enter to send · Shift+Enter for a new line
        </span>
        <span className="text-[10px] text-ink-faint">
          Answers follow the language you write in
        </span>
      </div>
    </div>
  )
}
