You are mapping Kalshi markets to equivalent markets on other platforms for ARBITRAGE detection and CONDITIONAL-CONSISTENCY checks.

# Input
- KALSHI_MARKETS: {kalshi_markets_json}
- OTHER_PLATFORM: {other_platform}
- OTHER_MARKETS: {other_markets_json}

# Task
For each Kalshi market, find the SINGLE most semantically equivalent market on OTHER_PLATFORM, if one exists.

Two markets are "equivalent" only if:
1. They resolve on the same underlying event,
2. On the same date/time window (within 24h),
3. With the same YES condition (no inverted polarity).

Return STRICT JSON list:
[
  {{
    "kalshi_ticker": "KXFED-25NOV",
    "other_ticker": "...",
    "equivalence": "exact" | "approximate" | "inverted" | "none",
    "polarity_flip": false,
    "confidence": 0.0-1.0,
    "notes": "one sentence on why"
  }},
  ...
]

Only emit "exact" or "approximate" mappings with confidence >= 0.8 for arbitrage use.
