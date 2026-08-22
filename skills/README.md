# AlphaMind Skills

At startup, AlphaMind loads Skills from `ALPHAMIND_SKILLS_DIR` and injects the matching one(s) into the corresponding agent's system prompt when a request matches. Skills are ideal for business-handling guidelines, research Q&A tone, compliance phrasing, suitability and escalation boundaries, and prohibitions.

Three built-in Skills:

```text
skills/market_info/SKILL.md            # Market & Information: quotes/indices/ETF info/term explanations, objective and neutral
skills/research_analysis/SKILL.md      # Research & Analysis: reports/financials/valuation/quant concepts, interpret only, no conclusions
skills/compliance_suitability/SKILL.md # Compliance & Suitability: risk ratings/disclosure/account/trading rules, compliance limits first
```

## Skill file format

Use one directory per Skill, with the main file named `SKILL.md`:

```text
skills/<skill_name>/SKILL.md
```

Add simple front matter at the top:

```markdown
---
name: Research & Analysis Guidelines
description: Guidelines for ResearchAgent covering reports, financials, valuation and quant concepts
keywords: 研报,财报,估值,research report,financial,valuation,factor
agents: research
enabled: true
---
```

Field notes:

- `name`: display name, shown in the injected prompt.
- `description`: short description, handy for the `/skills` endpoint.
- `keywords`: trigger keywords (Chinese and/or English); injected only when the user message matches one. Comma-separated.
- `agents`: applicable agents — `market`, `research`, `compliance` (comma-separated for multiple).
- `enabled`: `true` / `false`.

## Authoring guidelines

- Put important rules near the top; overly long content is truncated by the prompt budget.
- One Skill describes one responsibility; do not mix market, research and compliance rules in one file.
- Include stable sections such as "Role", "Handling", "Escalation" and "Prohibited".
- For passwords, verification codes, account numbers and other sensitive data, state clearly that they must not be collected or disclosed.
- Use conservative wording for anything you cannot guarantee ("usually", "expected", "subject to verification").

## Hot reload

After editing a Skill, no restart is needed — call:

```bash
curl -X POST http://localhost:8000/skills/reload
```

Check the load result and parse errors:

```bash
curl http://localhost:8000/skills
```
