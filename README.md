# Portfolio Intelligence Platform

A production-style portfolio analytics platform: ingestion → PostgreSQL
warehouse → dbt marts → Python analytics engine → Monte Carlo
simulation → ML volatility-regime classifier with MLflow tracking →
FastAPI backend → React/TypeScript frontend.

## Known Limitations

### ML volatility-regime classifier: chronological train/test split can straddle a genuine volatility regime change

The classifier's chronological 70/15/15 train/val/test split (on
~5 years of TCS.NS/RELIANCE.NS price history, 2021-09-27 to
2026-09-25) currently produces:

- train: 2021-12-28 → 2025-04-04 (24.8% positive labels)
- val: 2025-04-07 → 2025-12-17 (9.2% positive labels)
- test: 2025-12-18 → 2026-08-27 (51.4% positive labels)

This skew was investigated in depth (see `ml/labels/labels.py`'s
module docstring for the full incident writeup) and is **not**
primarily a leakage bug. An earlier version of the label threshold
computation did leak test-period data into the training-period "high
volatility" bar, and that was fixed (the threshold is now computed
only from each ticker's own training-period volatility). Fixing it
barely moved the skew (test label rate went from 50.6% to 51.4%),
which is itself the evidence: **TCS.NS's realized volatility
genuinely rose in the test period** (quarterly mean realized-forward
volatility: 2025Q4 = 0.167, 2026Q1 = 0.288, 2026Q2 = 0.342,
2026Q3 = 0.311 — a real, sustained elevation, not a computation
artifact) while RELIANCE.NS's test-period volatility was unremarkable
relative to its own history. This is a real characteristic of the
underlying market data landing inside the chronological test window,
which a single train/test split cannot control for.

**We deliberately did not respond to this by picking a different date
range or split ratio that produces a more favorable test period.**
That would be cherry-picking the split to manufacture a promotion,
which this project's own dev rules prohibit. The promotion gate
correctly rejects the resulting model (F1 below baseline by a wide
margin on the primary split), and that rejection is reported as-is.

**What we did instead**: added walk-forward (rolling-origin)
cross-validation (`ml/walk_forward.py`) as a supplementary diagnostic,
surfaced on the ML Insights page alongside (never replacing) the
primary single-split evaluation and promotion gate. It shows the model
actually **beats** the persistence baseline in the two earliest folds
(delta F1 of +0.18 and +0.16) and **underperforms** in the three most
recent folds (delta F1 of -0.02, -0.14, and -0.36) — the model is not
uniformly bad, it specifically struggles in the same high-volatility
regime the primary test split happens to land in. This is a more
honest and more complete picture than either "the model works" or "the
model doesn't work" taken from a single split.

Walk-forward's scoping simplification (documented in
`ml/walk_forward.py`'s own module docstring): it reuses the same
per-ticker label threshold established once from the primary split's
70% training period, rather than re-deriving a fresh point-in-time
threshold per fold. A fully rigorous version would do the latter.

### ARIMA forecast: 80% prediction intervals are overconfident in backtesting

`forecasting/arima/` adds a complementary time-series forecast alongside
(not replacing) Monte Carlo — see `forecasting/__init__.py` for the
conceptual distinction between distribution-based simulation and
pattern-based forecasting. `auto_arima` selects order (0, 1, 0) on the
extended 5-year portfolio-value series — i.e. a driftless random walk,
not a discovered exploitable pattern, which is itself an honest and
plausible finding for asset-price-derived data.

A 5-origin walk-forward backtest (252-day horizon, no lookahead —
enforced and tested in `forecasting/tests/test_no_lookahead.py`) found:

- MAE ≈ 6,294 / RMSE ≈ 6,463 (on a series ranging ~20,300–37,100)
- **95% interval observed coverage: 100%** (5/5) — consistent with, if
  not more conservative than, the stated confidence level
- **80% interval observed coverage: 60%** (3/5) — below the stated 80%
  confidence, i.e. **this model's 80% intervals are overconfident** in
  this backtest

We are reporting this plainly rather than hiding it or dropping the 80%
interval from the output, per this project's own dev rules: an honest
uncertainty estimate that turns out imperfect is more valuable than a
falsely reassuring one. The important caveat in the other direction:
n=5 origins is a small sample (the 252-day horizon combined with 5 years
of history leaves little room for more independent, non-overlapping
origins), so 60% vs. a stated 80% is suggestive of overconfidence rather
than a tight, statistically conclusive result on its own — both the
finding and this caveat about sample size are surfaced together in the
API response and the Forecasting page's UI.

### Invested Capital may differ slightly from the broker-reported figure

For a real uploaded portfolio (a converted Groww "Stocks Order History"
export), Invested Capital lands close to but not exactly at the
broker's own reported figure (confirmed real case: platform ₹15,696.25
vs. Groww's real ₹15,769.39 — a 0.46% gap; Market Value matched almost
exactly, ₹17,371.54 vs. ₹17,371.62). Investigated directly against the
real source files, not assumed:

- The raw order-history export's own "Value" column is confirmed pure
  `quantity × price` for every row (checked by hand against all 54
  real rows, zero exceptions) — it does not include brokerage/STT/
  other transaction charges, and the file has no separate charges
  column at all.
- Real per-trade charges DO exist and ARE recoverable, but only from a
  SEPARATE export (the Groww "Balance Statement" `STOCKS_SETTLEMENT`
  entries) cross-referenced by date against each order — confirmed
  directly: e.g. a real BUY with raw value ₹711.40 settled for a real
  debit of ₹714.78 (a ₹3.38, 0.475% charge). This works cleanly for
  the 26 of 32 real order-dates where a settlement day contains only
  BUYs or only SELLs.
- Applying these recovered charges to Invested Capital was tried and
  measured, not assumed to help: every variant tested (an averaged
  fallback rate for ambiguous mixed-settlement days; a conservative
  exact-match-only version) made the gap against the real ground truth
  LARGER, not smaller (up to +2.6%, vs. the real -0.46% without any fee
  adjustment at all). This is itself informative: it suggests Groww's
  own displayed "Invested" figure is computed from raw principal
  (quantity × price) without folding in transaction charges — a common
  brokerage convention — so applying recovered charges here would be
  correcting for a difference that Groww's own number doesn't correct
  for either.
- One specific real trade (`HYUNDAI.NS`, 73% of this portfolio's
  weight) has NO matching settlement entry at all on its execution
  date (2024-10-21) — the real date of Hyundai Motor India's NSE IPO
  listing. An IPO allotment settles via ASBA/UPI mandate, not a normal
  T+1 stock settlement, and Indian brokers typically charge no
  brokerage on IPO applications, so this is consistent with a real
  zero-fee trade rather than a data gap.

Given the above, fees/tax are left at 0 for every converted transaction
(matching, not working around, how Groww's own "Invested" figure
appears to be computed), and the small remaining gap is treated as an
accepted, explained limitation of the available export data rather than
something to keep chasing.
