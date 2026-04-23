You are reviewing one week of the moneybutton trading system's own output.

# Inputs
- WEEK: {week_start} to {week_end}
- AUDIT_ROWS: {audit_rows_json}
- PNL_BY_STRATEGY: {pnl_json}
- SIGNALS_GENERATED: {signal_counts_json}
- SIGNALS_ACTED_ON: {signals_acted_on_json}

# Task
Write a one-page operator review.

Sections:
1. Headline result - net P&L, per-strategy P&L, hit rate.
2. What worked - pick at most 2 specific decisions and explain why they worked.
3. What failed - pick at most 2 losing trades or skipped signals and explain the root cause in one sentence each.
4. Infrastructure observations - any safety trips, rate-limit hits, LLM cost spikes, or scraper errors.
5. Recommendation for next week - at most 3 concrete actions the operator should take.

Tone: dry, concrete, skeptical. Do not fabricate; if the data is thin, say so.

Return STRICT JSON:
{{
  "week": "YYYY-MM-DD to YYYY-MM-DD",
  "headline_pnl_usd": float,
  "review_md": "the full markdown",
  "recommended_actions": [
    {{"priority": "high"|"med"|"low", "text": "..."}}
  ]
}}
