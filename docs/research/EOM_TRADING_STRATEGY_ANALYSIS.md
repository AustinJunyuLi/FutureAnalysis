# EOM Trading Strategy Analysis: S1 Calendar Spread

**Date**: 2025-11-28
**Strategy**: Buy S1 spread at EOM-5, Sell at EOM-1

---

## Strategy Definition

- **Entry**: Buy S1 spread (long F2, short F1) at EOM-4 (5th to last trading day) at market open
- **Exit**: Sell S1 spread at EOM-1 (2nd to last trading day) at market close
- **Holding period**: ~4 trading days (3 price intervals)
- **Underlying**: CME High Grade Copper (HG) calendar spread

---

## Verdict: YES, Profitable on Average (with caveats)

The strategy exploits the documented month-end roll pressure effect where institutional investors rolling from F1 to F2 systematically widen the S1 spread during the final 5 trading days of each month.

---

## Expected Returns

| Metric | Value |
|--------|-------|
| **Average daily S1 widening** | +0.212 cents/lb |
| **Holding period** | 3 intervals (EOM-4 to EOM-1) |
| **Expected gross profit** | ~0.64 cents/lb |
| **Per contract (25,000 lbs)** | **~$160** |
| **Win rate** | ~75% of months |
| **Statistical significance** | p < 0.001 |

### After Transaction Costs

| Cost Component | Estimate |
|----------------|----------|
| Bid-ask spread (round-trip) | 0.10-0.20 cents |
| Slippage + commissions | ~0.05 cents |
| **Total costs** | ~0.15-0.25 cents |
| **Net profit per contract** | **~$100-125** |

---

## Critical Issue: Missing the Best Days

The EOM effect has **intensified over time** and is now **concentrated in the final two days**:

| Day | Time Trend (cents/year) | p-value | Status |
|-----|------------------------|---------|--------|
| EOM-4 | -0.075 | 0.075 | No significant trend |
| EOM-3 | +0.013 | 0.721 | No significant trend |
| EOM-2 | +0.020 | 0.602 | No significant trend |
| **EOM-1** | **+0.079** | **0.005** | **Strengthening** |
| **EOM** | **+0.074** | **0.016** | **Strengthening** |

**By exiting at EOM-1 close, you capture EOM-1 but miss the EOM day entirely.**

---

## Strategy Comparison

| Strategy | Entry | Exit | Gross $/contract | Net after costs | Win Rate |
|----------|-------|------|------------------|-----------------|----------|
| **Your plan** | EOM-4 open | EOM-1 close | ~$160 | ~$100-125 | ~75% |
| Full window | EOM-4 open | EOM close | ~$212 | ~$150-175 | ~76% |
| Last 2 days only | EOM-2 open | EOM close | ~$100 | ~$50-75 | ~65% |

---

## Recommended Improvement

**Hold through EOM (last trading day) for ~30% more profit:**

- Entry: EOM-4 (beginning of day)
- Exit: EOM (end of last trading day)
- Captures full 5-day window: **0.85 cents = $212.50/contract**

---

## Historical Context

- **Sample period**: 2015-2024 (119 months)
- **Effect growth**: 0.08 cents/day (2015) to 0.63 cents/day (2024) - **8x increase**
- **Economic driver**: Passive index fund roll activity + liquidity fragmentation near expiry
- **Per-contract cost of rolling at month-end**: $212.50 (this is the alpha source)

---

## Risk Factors

1. **Transaction costs** consume ~40-50% of gross profit
2. **25% of months are losers** - position sizing matters
3. **Effect may diminish** if more traders exploit it
4. **Basis risk** - spread can widen further before reverting
5. **Margin requirements** for spread positions

---

## Data Sources

- Seasonality report: `presentation_docs/seasonality_report.tex`
- Underlying data: `outputs/s1_eom_trend_2015_2024/eom_data_long.csv`
- Analysis scripts: `scripts/s1_eom_trend.py`, `scripts/s1_eom_intra_month_slope.py`

---

## Next Steps

1. Backtest exact entry/exit timing (open vs close prices)
2. Analyze year-by-year returns for strategy stability
3. Calculate Sharpe ratio and max drawdown
4. Test optimal position sizing (Kelly criterion)
5. Evaluate extending hold to EOM vs current EOM-1 exit
