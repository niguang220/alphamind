import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import Markdown from './Markdown'

/**
 * This renderer parses untrusted model output. The cases that matter are the ones where a
 * malformed or hostile document could break the parser or escape into the DOM.
 */
describe('Markdown', () => {
  it('renders headings, bold and inline code without markup leaking through', () => {
    const { container } = render(
      <Markdown text={'## Valuation\n\nA **P/E** of `30` is high.'} />,
    )
    expect(screen.getByText('Valuation')).toBeTruthy()
    expect(container.querySelector('strong')?.textContent).toBe('P/E')
    expect(container.querySelector('code')?.textContent).toBe('30')
    expect(container.textContent).not.toContain('**')
    expect(container.textContent).not.toContain('##')
  })

  it('treats HTML in model output as text, never as markup', () => {
    const { container } = render(
      <Markdown text={'<img src=x onerror="alert(1)"> and <script>alert(2)</script>'} />,
    )
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
    expect(container.textContent).toContain('<script>')
  })

  it('renders a table with a header row', () => {
    const { container } = render(
      <Markdown text={'| Metric | Value |\n|---|---|\n| P/E | 30 |\n| ROE | 25% |'} />,
    )
    expect(container.querySelectorAll('th')).toHaveLength(2)
    expect(container.querySelectorAll('tbody tr')).toHaveLength(2)
    expect(screen.getByText('Metric')).toBeTruthy()
  })

  it('keeps ordered and unordered lists distinct', () => {
    const { container } = render(<Markdown text={'1. first\n2. second'} />)
    expect(container.textContent).toContain('1.')
    expect(container.textContent).toContain('second')
  })

  it('survives a table whose rows have ragged cell counts', () => {
    expect(() =>
      render(<Markdown text={'| A | B | C |\n|---|---|---|\n| 1 |\n| 1 | 2 | 3 | 4 |'} />),
    ).not.toThrow()
  })

  it('survives an unterminated table and a stray separator', () => {
    expect(() => render(<Markdown text={'|---|---|\n| dangling'} />)).not.toThrow()
    expect(() => render(<Markdown text={'| head |\n|---|'} />)).not.toThrow()
  })

  it('handles empty, whitespace and undefined input', () => {
    for (const value of ['', '   \n\n  ', undefined, null]) {
      expect(() => render(<Markdown text={value} />)).not.toThrow()
    }
  })

  it('does not hang on unbalanced emphasis markers', () => {
    const { container } = render(<Markdown text={'**unclosed and `unclosed'} />)
    expect(container.textContent).toContain('unclosed')
  })
})
