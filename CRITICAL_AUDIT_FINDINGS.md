# Critical Audit of Multi-Spread Detection Methodology

## Executive Summary

After a meticulous examination of the multi-spread detection framework, I've identified several **critical methodological issues** that fundamentally undermine the validity of the results. While the implementation is technically correct, the underlying approach contains serious flaws that lead to **spurious pattern detection** and **misinterpretation of market dynamics**.

## Critical Finding #1: Timing Analysis Misattribution ⚠️

### The Problem
The code measures "days to expiry" using the **front contract of each spread** at the event time:
- For S1 events: measures days until F1 expires
- For S2 events: measures days until F2 expires
- For S3 events: measures days until F3 expires

### Why This Is Wrong
This creates a **tautological relationship**. The pattern appears to "ripple through" because:
1. S1 = F2 - F1 (spread between first and second month)
2. S2 = F3 - F2 (spread between second and third month)
3. When F1 is 28 days from expiry, F2 is ~59 days from expiry, F3 is ~90 days from expiry

The "consistent 28-30 day pattern" is an **artifact of the measurement method**, not a real market phenomenon.

### What Should Be Done Instead
All spreads should be measured relative to a **common reference point**:
- Option 1: Days until F1 expires (for all spreads)
- Option 2: Days until the nearest contract expires
- Option 3: Calendar date analysis without expiry-relative timing

## Critical Finding #2: Z-Score Detection on Non-Stationary Data ⚠️

### The Problem
The code uses rolling z-scores with parameters:
```python
window=20  # buckets (~2 days)
z_threshold=1.5
```

### Why This Is Wrong
1. **Non-stationarity**: Futures spreads exhibit strong seasonality and term structure effects
2. **Window too short**: 20 buckets (~2 days) cannot capture the monthly roll cycle
3. **No detrending**: The method doesn't account for systematic drift in spreads as contracts mature
4. **False positive rate**: With z=1.5, we expect ~6.7% false positives even in random noise

### Evidence of the Problem
- S1: 6.16% detection rate
- S2: 5.81% detection rate
- S3: 5.11% detection rate

These rates are **suspiciously close to the statistical false positive rate** for z=1.5 threshold.

## Critical Finding #3: Survivorship and Selection Bias ⚠️

### The Problem
The contract identification always selects the "surviving" contracts:
```python
delta[~active_mask] = np.inf  # Excludes expired/inactive contracts
```

### Why This Is Wrong
1. **Survivorship bias**: Only analyzes contracts that haven't expired yet
2. **Look-ahead bias**: Contract selection uses future expiry information
3. **Dynamic universe**: The set of available contracts changes over time, creating artificial patterns

### Impact
This creates systematic patterns around contract transitions that appear as "events" but are actually **structural artifacts** of the contract universe evolving.

## Critical Finding #4: Spread Calculation Issues ⚠️

### The Problem
Spreads are calculated as simple price differences:
```python
spread = f_i_plus_1_prices - f_i_prices
```

### Why This Is Wrong
1. **No normalization**: Raw price differences don't account for price level changes
2. **Missing data handling**: NaN values propagate through calculations
3. **No adjustment for contract specifications**: Different contract months may have different multipliers or conventions

### Recommended Fix
```python
# Better approach:
spread_pct = (f_i_plus_1_prices - f_i_prices) / f_i_prices
# Or use log returns:
spread_log = np.log(f_i_plus_1_prices) - np.log(f_i_prices)
```

## Critical Finding #5: Cool-Down Period Masks True Frequency ⚠️

### The Problem
3-hour cool-down period suppresses multiple events:
```python
cool_down_hours: 3.0
```

### Why This Is Wrong
1. **Information loss**: Genuine market events may occur in clusters
2. **Arbitrary threshold**: No theoretical justification for 3 hours
3. **Biases timing analysis**: Events get "pushed forward" to after cool-down expires

## Critical Finding #6: Circular Logic in Hypothesis Testing ⚠️

### The Core Issue
The analysis claims to test whether patterns are "institutional" vs "mechanical", but:

1. **Hypothesis A** (Institutional): Traders roll at specific times
2. **Hypothesis B** (Mechanical): Patterns reflect expiry mechanics

### The Flaw
**Both hypotheses predict the same observable pattern** - events occurring before expiry. The test cannot distinguish between them because:
- If institutions roll 28 days before expiry → we see events at 28 days
- If mechanics cause widening 28 days before expiry → we see events at 28 days

The multi-spread analysis doesn't resolve this - it just confirms that **something happens** consistently, not **why** it happens.

## Critical Finding #7: Statistical Validity Issues ⚠️

### Multiple Testing Problem
- Testing 11 spreads simultaneously without correction
- P-values and significance levels not adjusted for multiple comparisons
- Risk of finding spurious patterns by chance alone

### Correlation Interpretation
Low correlations (0.05-0.32) between spreads are interpreted as "independence", but this ignores:
- Non-linear relationships
- Lag effects (correlations at different time offsets)
- Common underlying factors (interest rates, seasonality)

## Recommendations

### Immediate Actions Required

1. **Fix Timing Reference**
   ```python
   # Measure all spreads relative to F1 expiry
   days_to_f1_expiry = (expiry_map[contract_chain['F1']] - event_date).days
   ```

2. **Implement Proper Statistical Tests**
   - Use GARCH models for volatility clustering
   - Apply Bonferroni correction for multiple testing
   - Test for structural breaks around known expiry dates

3. **Add Baseline Comparisons**
   - Generate synthetic data with known properties
   - Compare detection rates against random walk simulations
   - Use shuffle tests to establish null hypothesis baselines

4. **Improve Event Detection**
   ```python
   # Use percentage changes and longer windows
   spread_pct = spread.pct_change()
   window = 150  # ~15 days for proper context
   ```

5. **Document Assumptions**
   - Clearly state what constitutes an "event"
   - Define success criteria before analysis
   - Acknowledge limitations and biases

### Fundamental Redesign Needed

The current approach cannot definitively answer whether patterns are institutional or mechanical. A proper analysis requires:

1. **Control group**: Analyze non-deliverable forwards or cash-settled futures
2. **Natural experiments**: Study rule changes or market disruptions
3. **Microstructure analysis**: Examine order flow and trade sizes
4. **Cross-market validation**: Test on different commodities/exchanges

## Conclusion

While the implementation is technically competent, the methodology contains **fundamental flaws** that invalidate the conclusions. The detected patterns are likely **statistical artifacts** rather than genuine market behaviors. The analysis suffers from:

- Tautological timing measurements
- Inappropriate statistical methods for non-stationary data
- Multiple forms of selection bias
- Circular reasoning in hypothesis testing

**The current results should not be used for trading decisions or academic publication** without addressing these critical issues. The entire analytical framework needs redesign with proper statistical rigor and clear, testable hypotheses.

## Severity Assessment

🔴 **CRITICAL**: Timing analysis methodology - renders all timing conclusions invalid
🔴 **CRITICAL**: Non-stationary z-score detection - produces spurious events
🟠 **HIGH**: Selection and survivorship biases - distorts pattern interpretation
🟡 **MEDIUM**: Spread calculation issues - affects magnitude but not existence of patterns
🟡 **MEDIUM**: Cool-down and multiple testing - masks true statistical properties

---

*Generated by critical audit on November 4, 2025*