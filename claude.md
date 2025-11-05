# Futures Roll Analysis - Development Log

## Project Status (Updated 2025-11-05)

### Current State

The framework successfully implements multi-spread comparative analysis across 11 calendar spreads (S1-S11) for CME copper futures. The analysis definitively demonstrates that detected spread widening events at 28-30 days before contract expiry reflect **systematic contract expiry mechanics** rather than discretionary institutional rolling behavior.

**Key Evidence:**
- S1, S2, S3 exhibit similar event rates (6.16%, 5.81%, 5.11%)
- All spreads show events ~28-30 days before their respective front contract expiry
- Low inter-spread correlations (0.05-0.32) indicate independent, contract-specific dynamics
- Pattern "ripples through" contract chain as successive contracts mature

### Critical Finding

The original hypothesis that spread widening indicates **institutional roll timing** has been disproven. Instead, the patterns represent **universal contract maturity effects** driven by:
- Liquidity migration from expiring to next-month contracts
- Open interest decline as positions close or roll
- Convergence dynamics as spot and futures prices align
- Market maker inventory adjustments ahead of delivery

## Plan Moving Forward: Identifying TRUE Institutional Rolling

To distinguish genuine institutional rolling activity from structural expiry mechanics, implement the following techniques:

### Phase 1: Multi-Dimensional Signal Analysis

#### 1. Cross-Spread Confirmation
**Objective:** Require simultaneous signals across multiple spreads

**Implementation:**
- Detect when S1 AND S2 both widen within same time window
- Look for cascading effects across S1→S2→S3 (institutional rolls ripple through curve)
- True rolls show F2 volume rising AS F1 volume falls
- **Rationale:** Institutional rolls move size across the curve, not just front/next

**Code additions:**
```python
def detect_cross_spread_events(multi_spreads, multi_events):
    """Identify events where multiple adjacent spreads signal simultaneously"""
    # S1+S2 joint signals
    # S1+S2+S3 cascades
    # Time-windowed co-occurrence
```

#### 2. Flow/Volume Microstructure Analysis
**Objective:** Track actual position transfers between contracts

**Metrics to implement:**
- **Volume ratios:** F2_vol / F1_vol over rolling windows
- **VWAP crossover:** When F2 VWAP consistently exceeds F1 VWAP
- **Open Interest delta:** Track OI changes in F1 vs F2
- **Trade size distribution:** Institutional trades have larger average sizes
- **Time-of-day patterns:** Many institutions roll during specific windows (London close, NYMEX settlement)

**Required data:**
- Minute-level volume per contract (already have)
- Open Interest data (need to add to ingestion pipeline)
- Trade-level data if available (optional enhancement)

**Code additions:**
```python
def compute_volume_ratio_signal(panel, front_next):
    """Detect persistent F2 volume > F1 volume"""

def compute_oi_delta(panel, front_next):
    """Track open interest migration from F1 to F2"""

def detect_vwap_crossover(panel, front_next):
    """Identify when F2 VWAP crosses above F1 VWAP"""
```

#### 3. Calendar-Awareness: Index Roll Schedules
**Objective:** Filter for known institutional roll windows

**Key Dates to Track:**
- **Goldman Sachs Commodity Index (GSCI):** Rolls 5th-9th business day of month
- **Bloomberg Commodity Index (BCOM):** Different roll schedule
- **MSCI Commodity Indices:** Specific roll periods
- **Large ETF rebalancing:** USO, GLD, CORN, WEAT roll schedules

**Implementation:**
- Create calendar file: `metadata/calendars/index_roll_schedules.csv`
- Tag events occurring during known roll windows
- Compute enrichment: (events during roll window) / (expected based on time)

**Code additions:**
```python
def load_index_roll_calendar(calendar_path):
    """Load GSCI/BCOM/MSCI roll schedules"""

def tag_index_roll_periods(events, index_calendar):
    """Identify events occurring during known institutional roll windows"""

def compute_roll_window_enrichment(events, index_calendar):
    """Calculate if events cluster during index roll periods"""
```

#### 4. Comparative Clustering & Pattern Recognition
**Objective:** Use unsupervised learning to find consistent institutional signatures

**Features for clustering:**
- Spread volatility (S1, S2, S3)
- Volume ratios and deltas
- Time-to-expiry buckets
- Time-of-day (session)
- Bid-ask spreads (if available)
- Days-to-expiry when event occurs

**Methodology:**
- Apply K-means or DBSCAN clustering on event features
- Identify recurrent "motifs" (daily patterns, spread sequences)
- True institutional rolls should form distinct clusters
- Events not matching clusters are likely noise/expiry effects

**Validation approach:**
- Need labeled data: known institutional roll dates from CFTC reports or index schedules
- Train classifier on labeled data
- Validate cluster assignments

**Code additions:**
```python
def extract_event_features(multi_spreads, multi_events, panel):
    """Create feature matrix for clustering"""

def cluster_event_patterns(features, method='kmeans', n_clusters=5):
    """Apply unsupervised clustering to identify motifs"""

def validate_clusters_against_labeled_data(clusters, known_roll_dates):
    """Measure clustering accuracy using known institutional rolls"""
```

### Phase 2: External Data Integration

#### 5. Cross-Asset Corroboration
**Objective:** Align futures signals with related market activity

**Data sources to integrate:**
- **Options market:** Calendar spread options volume spikes
- **ETF flows:** USO, GLD, CORN, WEAT share creations/redemptions
- **FX hedges:** For commodities priced in USD, check for FX hedge adjustments
- **Cash market:** Physical inventory data (if available)
- **Term structure:** Watch entire curve, not just nearby contracts

**Implementation priority:**
1. Options volume (if accessible via API)
2. ETF flow data (public via ETF websites)
3. FX correlation analysis (can use free data sources)

**Code additions:**
```python
def load_options_volume(symbol, date_range):
    """Import calendar spread option volumes"""

def load_etf_flows(etf_symbol, date_range):
    """Import ETF creation/redemption data"""

def correlate_external_signals(futures_events, external_data):
    """Identify temporal alignment between markets"""
```

#### 6. CFTC Disaggregated Commitments of Traders (COT) Data
**Objective:** Match spread events to reported institutional position changes

**Data source:**
- CFTC releases Disaggregated COT every Friday (Tuesday data)
- Shows "Managed Money" (hedge funds, CTAs) vs "Producer/Merchant" vs "Swap Dealers"
- Available via: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

**Methodology:**
- Download historical COT data for copper (HG)
- Track weekly changes in Managed Money net positions
- Identify weeks with large position changes (>5% or >10%)
- Check if spread events cluster around those weeks

**Known limitations:**
- 3-day publication lag (Tuesday → Friday)
- Weekly granularity (vs our intraday/daily analysis)
- Aggregated across all contracts (can't see F1→F2 specifically)

**Code additions:**
```python
def download_cot_data(commodity='HG', start_year=2008, end_year=2024):
    """Fetch CFTC COT reports via API or scraping"""

def compute_managed_money_delta(cot_data):
    """Calculate week-over-week changes in institutional positions"""

def align_events_to_cot_changes(events, cot_deltas, threshold_pct=5):
    """Identify if events coincide with large COT position shifts"""
```

### Phase 3: Integrated Roll Detection Pipeline

**Goal:** Combine all signals into a composite "institutional roll likelihood" score

**Scoring system:**
```python
def compute_institutional_roll_score(event):
    score = 0

    # +2 points: Occurs during GSCI/BCOM roll window
    if event in index_roll_window:
        score += 2

    # +2 points: S1 AND S2 widen simultaneously
    if cross_spread_signal:
        score += 2

    # +1 point: F2 volume > 80% of F1 volume for 3+ consecutive sessions
    if persistent_volume_flip:
        score += 1

    # +1 point: OI delta shows F1 declining AND F2 rising
    if oi_migration:
        score += 1

    # +1 point: COT report shows large managed money position change
    if cot_signal:
        score += 1

    # +1 point: Event belongs to recurrent cluster identified by ML
    if in_institutional_cluster:
        score += 1

    # -2 points: Event within 5 days of expiry (likely expiry noise)
    if days_to_expiry <= 5:
        score -= 2

    return score  # Score 0-8, threshold ≥5 for "likely institutional"
```

### Implementation Roadmap

**Week 1-2: Calendar & Volume Analysis**
- [ ] Create `metadata/calendars/index_roll_schedules.csv` with GSCI/BCOM dates
- [ ] Implement cross-spread confirmation logic
- [ ] Add volume ratio and OI delta calculations
- [ ] Tag events by roll window

**Week 3-4: Clustering & Pattern Recognition**
- [ ] Extract event feature matrix
- [ ] Implement K-means clustering
- [ ] Validate against known roll dates

**Week 5-6: External Data Integration**
- [ ] Download CFTC COT historical data
- [ ] Align COT position changes with events
- [ ] (Optional) Add ETF flow data if accessible

**Week 7-8: Integrated Scoring & Validation**
- [ ] Build composite roll likelihood score
- [ ] Backtest on 2008-2024 dataset
- [ ] Generate "high-confidence institutional roll" dataset
- [ ] Compare to baseline (current expiry-based events)

### Success Metrics

1. **Precision:** % of flagged events that align with known institutional rolls
2. **Enrichment:** Ratio of events during roll windows vs random expectation
3. **Clustering validity:** Silhouette score >0.5 for identified patterns
4. **External validation:** Correlation with COT position changes >0.3

### Files to Create/Modify

**New files:**
- `src/futures_roll_analysis/multi_spread_analysis.py` - ✅ DONE
- `src/futures_roll_analysis/volume_microstructure.py` - To add
- `src/futures_roll_analysis/external_data.py` - To add
- `src/futures_roll_analysis/clustering.py` - To add
- `metadata/calendars/index_roll_schedules.csv` - To create

**Modified files:**
- `src/futures_roll_analysis/analysis.py` - Add new pipeline stages
- `src/futures_roll_analysis/events.py` - Add composite scoring
- `config/settings.yaml` - Add new configuration sections

## References

**Index Roll Schedules:**
- GSCI Methodology: https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-gsci.pdf
- Bloomberg Commodity Index: https://www.bloomberg.com/professional/product/indices/

**CFTC Data:**
- COT Reports: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- Legacy vs Disaggregated: Use Disaggregated for "Managed Money" visibility

**Academic Literature:**
- Erb & Harvey (2006): "The Strategic and Tactical Value of Commodity Futures"
- Mou (2011): "Limits to Arbitrage and Commodity Index Investment"
- Henderson et al (2015): "New Evidence on the Financialization of Commodity Markets"

## Notes

- Current implementation correctly identifies expiry-driven events
- These are NOT institutional rolling decisions but universal market mechanics
- Future work must look BEYOND simple spread widening
- Success requires multi-dimensional analysis combining spread, volume, timing, and external signals
