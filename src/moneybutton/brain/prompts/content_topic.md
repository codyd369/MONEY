You pick ONE AI developer tool to write an article about today.

# Inputs
- TRENDING: {trending}
- RECENTLY_COVERED: {recently_covered}
- DATE: {date}

# Task
Select the single best topic from TRENDING that:
1. Is a concrete AI developer tool (CLI, IDE extension, SDK, API, agent framework, or eval harness).
2. Is not in RECENTLY_COVERED.
3. Has enough public information (docs, pricing page, 3+ third-party reviews) that a 900-word article can be written without fabrication.

Return STRICT JSON:
{{
  "tool_slug": "kebab-case-slug",
  "tool_name": "Display Name",
  "source_url": "https://...",
  "why_now": "one sentence - why this tool, today",
  "risk": "what could be wrong about this pick (info too thin, etc.)"
}}
