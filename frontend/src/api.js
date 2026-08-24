// Thin client for the AlphaMind backend.
// Dev goes through the Vite proxy (/api -> :8000) so the browser stays same-origin.
// No credentials ever live here: the API key is backend-only, in its .env.

const BASE = import.meta.env.VITE_ALPHAMIND_API ?? '/api'

async function request(path, { method = 'GET', body, timeoutMs = 120000 } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { 'content-type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}`)
    }
    return await res.json()
  } finally {
    clearTimeout(timer)
  }
}

export const health = () => request('/health', { timeoutMs: 4000 })

export const chat = ({ message, userId, convId }) =>
  request('/chat', {
    method: 'POST',
    body: { message, user_id: userId, conv_id: convId ?? null },
  })

/**
 * A guardrail interception is NOT the same thing as `escalated`.
 *
 * A domain agent can also set escalated=true when its own answer offers to hand
 * off to a human advisor (see BaseAgent._needs_escalation) — a Compliance answer
 * about margin trading does exactly that. The unambiguous marker for the
 * investment-advice guardrail is the routing reason the orchestrator stamps on
 * the short-circuited result.
 */
export const isGuardrail = (turn) =>
  typeof turn?.routing_reason === 'string' && turn.routing_reason.startsWith('guardrail=')
