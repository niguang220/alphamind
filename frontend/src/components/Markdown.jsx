import { Fragment } from 'react'

/**
 * Small markdown renderer for agent answers.
 *
 * Deliberately builds React elements instead of setting innerHTML: model output is
 * untrusted text, and this keeps the XSS surface at zero with no extra dependency.
 * Supports what the agents actually emit — headings, tables, lists, rules, bold,
 * inline code — and falls through to paragraphs for everything else.
 */

function inline(text, keyBase) {
  const out = []
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g
  let last = 0
  let m
  let i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const tok = m[0]
    if (tok.startsWith('**')) {
      out.push(
        <strong key={`${keyBase}-b${i}`} className="font-semibold text-ink">
          {tok.slice(2, -2)}
        </strong>,
      )
    } else {
      out.push(
        <code
          key={`${keyBase}-c${i}`}
          className="border border-hairline bg-raised px-1 py-[1px] font-mono text-[0.9em] text-ink"
        >
          {tok.slice(1, -1)}
        </code>,
      )
    }
    last = m.index + tok.length
    i += 1
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

const splitRow = (line) =>
  line
    .trim()
    .replace(/^\||\|$/g, '')
    .split('|')
    .map((c) => c.trim())

export default function Markdown({ text }) {
  const lines = (text || '').replace(/\r/g, '').split('\n')
  const blocks = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    if (!line.trim()) {
      i += 1
      continue
    }

    // horizontal rule
    if (/^\s*(-{3,}|_{3,}|\*{3,})\s*$/.test(line)) {
      blocks.push(<hr key={i} className="my-5 border-0 border-t border-rule" />)
      i += 1
      continue
    }

    // heading
    const h = line.match(/^(#{1,6})\s+(.*)$/)
    if (h) {
      const depth = h[1].length
      blocks.push(
        depth <= 2 ? (
          <h4
            key={i}
            className="mt-6 mb-2.5 flex items-center gap-2.5 font-mono text-meta font-semibold tracking-[0.16em] text-ink-soft uppercase first:mt-0"
          >
            <span className="h-px w-3 bg-rule-bright" />
            {h[2].replace(/[#*]/g, '')}
          </h4>
        ) : (
          <h5 key={i} className="mt-4 mb-1.5 text-ui font-semibold text-ink">
            {inline(h[2].replace(/[#*]/g, ''), `h${i}`)}
          </h5>
        ),
      )
      i += 1
      continue
    }

    // table
    if (line.trim().startsWith('|') && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] ?? '')) {
      const head = splitRow(line)
      const rows = []
      i += 2
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        rows.push(splitRow(lines[i]))
        i += 1
      }
      blocks.push(
        <div key={`t${i}`} className="my-4 overflow-x-auto">
          <table className="w-full border-collapse text-ui">
            <thead>
              <tr>
                {head.map((c, ci) => (
                  <th
                    key={ci}
                    className="border-b border-rule-bright px-2.5 py-2 text-left font-mono text-meta tracking-[0.08em] whitespace-nowrap text-ink uppercase"
                  >
                    {c.replace(/\*/g, '')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri} className="border-b border-hairline odd:bg-white/[0.012]">
                  {r.map((c, ci) => (
                    <td
                      key={ci}
                      className={`px-2.5 py-2 align-top ${ci === 0 ? 'text-ink-soft' : 'tabular text-ink-mute'}`}
                    >
                      {inline(c, `td${ri}-${ci}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }

    // lists
    const isUl = /^\s*[-*•]\s+/.test(line)
    const isOl = /^\s*\d+[.)]\s+/.test(line)
    if (isUl || isOl) {
      const items = []
      while (i < lines.length) {
        const l = lines[i]
        const mu = l.match(/^\s*[-*•]\s+(.*)$/)
        const mo = l.match(/^\s*(\d+)[.)]\s+(.*)$/)
        if (isUl && mu) items.push({ body: mu[1] })
        else if (isOl && mo) items.push({ n: mo[1], body: mo[2] })
        else break
        i += 1
      }
      blocks.push(
        <ul key={`l${i}`} className="my-3 space-y-1.5">
          {items.map((it, ii) => (
            <li key={ii} className="flex gap-2.5 text-ui leading-[1.62] text-ink-soft">
              <span className="tabular mt-[2px] w-4 shrink-0 text-right text-meta text-ink-faint">
                {isOl ? `${it.n}.` : '\u2022'}
              </span>
              <span className="min-w-0">{inline(it.body, `li${ii}`)}</span>
            </li>
          ))}
        </ul>,
      )
      continue
    }

    // paragraph
    const buf = []
    while (i < lines.length && lines[i].trim() && !/^\s*([-*•#>|]|\d+[.)])\s/.test(lines[i])) {
      buf.push(lines[i].trim())
      i += 1
    }
    if (buf.length) {
      blocks.push(
        <p key={`p${i}`} className="my-2.5 text-body leading-[1.68] text-ink-soft first:mt-0">
          {inline(buf.join(' '), `p${i}`)}
        </p>,
      )
    } else {
      i += 1
    }
  }

  return <Fragment>{blocks}</Fragment>
}
