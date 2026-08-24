# AlphaMind — Securities Research Desk (frontend)

A research-terminal UI for the AlphaMind backend. It is not a generic chat client: the
point is to make the orchestrator's reasoning legible — which agent took the request, why,
how confident the fused intent vote was, and where an investment-advice request got stopped.

## Run it

The backend must be running on `:8000` first (see the repository README).

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Dev requests go to `/api/*` and Vite proxies them to `http://localhost:8000`, so the browser
stays same-origin and CORS never enters the picture. Point it elsewhere with
`VITE_ALPHAMIND_API=https://host` if you need to.

**No credentials live here.** The model API key is backend-only, read from the backend's
`.env`. The frontend calls this project's own HTTP API and nothing else.

## What it shows

**Guardrail interception.** When the orchestrator short-circuits an advice request, the
answer renders as a compliance card — risk disclosure as a numbered list, an `escalated`
marker, and the routing reason that proves it was the guardrail. The pipeline strip below it
visibly stops at stage 2 with routing and generation struck out, because generation genuinely
never ran; the refusal is a fixed template, not model output.

A guardrail hit is detected by `routing_reason` starting with `guardrail=`, *not* by
`escalated`. A domain agent can also set `escalated` when its own answer offers a human
advisor — a Compliance answer about margin trading does exactly that — and rendering those as
guardrail cards would be wrong.

**Routing telemetry.** Every answer carries its agent, intent and group, primary and
supporting agents, routing reason and confidence, latency, and whether the knowledge base was
consulted. The inspector rail expands the latest turn; click any answer to pin it.

**Intent fusion, including what is switched off.** The three-route vote is drawn as three
meters. On an Anthropic-compatible third-party provider the embedding route is disabled at
construction and its weight is redistributed, so it is drawn as an explicitly disabled track
with the reason — a permanently empty bar with no explanation would just look broken.

**The pipeline strip** mirrors the four real stages of `AgentOrchestrator.run()`. While a
request is in flight it advances on indicative pacing; the only measured number shown is the
end-to-end latency the backend reports.

## Layout

```
src/
  api.js                  backend client + the guardrail discriminator
  constants.js            agent metadata, pipeline stages, sample prompts, eval snapshot
  App.jsx                 shell, conversation state, turn routing
  components/
    TopBar.jsx            wordmark, compliance strapline, backend status
    Composer.jsx          input
    GuardrailCard.jsx     the compliance interception card
    AnswerCard.jsx        normal agent answer with badge header
    Pipeline.jsx          four-stage orchestrator strip
    Inspector.jsx         routing telemetry rail + evaluation snapshot
    FusionBars.jsx        three-route intent vote
    Markdown.jsx          renderer for agent answers
    Bits.jsx              chips, meters, fields
```

`Markdown.jsx` builds React elements rather than setting `innerHTML`: agent output is
untrusted text, so this keeps the XSS surface at zero without pulling in a sanitizer.

## Notes

- UI copy is English. The input accepts any language and the backend answers in the language
  you wrote in, so the font stack carries a CJK fallback.
- The evaluation panel is a snapshot of the last `POST /eval/run`. A live run is a
  minute-long batch, not an interaction.
- Intended for local use. The backend has no auth or rate limiting, so do not expose it.
