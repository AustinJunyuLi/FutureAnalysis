# Futures Roll Analysis Framework

A comprehensive Python framework for analyzing institutional roll patterns in futures markets through statistical detection of calendar spread dynamics.

## 🎯 Overview

This project analyzes 16+ years of futures contract data to detect when market participants (ETFs, hedge funds, index funds) roll their positions from expiring contracts to next-month contracts. The framework uses statistical methods to identify calendar spread widening events that indicate institutional rolling activity.

### Key Question Answered
**When do market participants roll their futures positions relative to contract expiration?**

**Answer Found**: Median of **12 days before expiry** with heterogeneous strategies (no single "roll date")

## ✨ Key Features

- **Hourly Analysis**: Variable-granularity buckets with hourly precision during US trading hours
- **Data Processing**: Handles 13,548+ futures files across 32 commodity types  
- **Statistical Detection**: Enhanced z-score detection with 3.3x more events (2,737 vs 826)
- **Dynamic Contract ID**: Automated front/next month contract identification
- **Extensible Framework**: Easily analyze any commodity futures
- **Comprehensive Output**: Panel data, roll signals, and summary statistics

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize raw futures data (if not already done)
python organize_data.py

# 3. Run hourly analysis for copper
cd etf_roll_analysis
python scripts/hourly_analysis.py

# 4. View results in outputs/
# - panels/hourly_panel.parquet (hourly bucket data)
# - roll_signals/hourly_widening.csv (2,737 roll events)
# - analysis/bucket_summary.csv (statistics by bucket)
```

## 📁 Project Structure

```
futures_roll_analysis/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── organize_data.py             # Step 1: Organizes raw futures files
├── data_inventory.csv           # Generated: Maps all files to commodities
│
├── organized_data/              # Organized futures data by commodity
│   ├── copper/                  # 202 HG contracts (2008-2024)
│   ├── gold/                    # GC contracts
│   ├── silver/                  # SI contracts
│   ├── crude_oil/               # CL contracts
│   └── [28 other commodities]   # Total: 32 commodity types
│
├── etf_roll_analysis/           # Main analysis framework
│   ├── README.md                # Framework documentation
│   ├── USER_GUIDE.md            # Usage examples and tutorials
│   │
│   ├── config/
│   │   └── settings.yaml        # Configuration parameters
│   │
│   ├── src/roll_analysis/       # Core Python modules
│   │   ├── ingest.py            # Data loading and bucket aggregation
│   │   ├── bucket.py            # Variable-granularity bucketing
│   │   ├── bucket_panel.py      # Bucket panel assembly
│   │   ├── bucket_events.py     # Bucket event detection
│   │   ├── data_quality.py      # Contract filtering
│   │   ├── rolls.py             # Front/next identification
│   │   ├── spread.py            # Calendar spread calculation
│   │   └── events.py            # Daily event detection
│   │
│   ├── scripts/                 # Analysis execution scripts
│   │   ├── hourly_analysis.py   # Main hourly bucket pipeline
│   │   ├── bucket_analysis.py   # Full bucket analysis
│   │   └── hg_analysis.py       # Daily analysis (for comparison)
│   │
│   ├── metadata/
│   │   └── contracts_metadata.csv  # Official CME expiry dates
│   │
│   └── outputs/                 # Generated results
│       ├── panels/              # Panel data files
│       │   ├── hourly_panel.parquet   # Hourly bucket data
│       │   └── hg_panel_simple.csv    # Daily price panel
│       ├── roll_signals/        # Detection results
│       │   ├── hourly_widening.csv    # 2,737 hourly roll events
│       │   ├── hg_spread.csv          # Calendar spreads
│       │   └── hg_liquidity_roll.csv  # Volume signals
│       └── analysis/            # Statistical summaries
│           ├── bucket_summary.csv     # Events by bucket
│           └── preference_scores.csv  # Volume-adjusted scores
│
└── presentation_docs/           # LaTeX reports and documentation
    ├── analysis_report.pdf      # Technical report (20 pages)
    └── README.md                # Report compilation guide
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Anaconda (recommended) or pip

### Setup Steps

```bash
# 1. Clone or download this repository
cd futures_individual_contracts_1min

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas, numpy, yaml; print('Dependencies OK')"
```

## 📊 Usage Guide

### Step 1: Organize Raw Data
```bash
python organize_data.py
```
This creates the `organized_data/` directory with 32 commodity subdirectories.

### Step 2: Configure Analysis
Edit `etf_roll_analysis/config/settings.yaml`:
```yaml
data:
  minute_root: "../organized_data/copper"  # Change commodity here
  timezone: "US/Central"
  price_field: "close"

spread:
  method: "zscore"
  window: 20           # Rolling window (days)
  z_threshold: 1.5     # Detection sensitivity
  cool_down: 3         # Days between events
```

### Step 3: Run Analysis
```bash
cd etf_roll_analysis
python scripts/hg_analysis.py
```

### Step 4: Interpret Results

**Key Output Files:**
- `panels/hg_panel_simple.csv`: Daily price data for all contracts
- `roll_signals/hg_widening.csv`: Boolean series (TRUE = roll event detected)
- `ANALYSIS_SUMMARY_UPDATED.md`: Summary statistics

**Understanding Results:**
```python
import pandas as pd

# Load roll events
widening = pd.read_csv('outputs/roll_signals/hg_widening.csv', 
                       index_col=0, parse_dates=True)

# Count roll events
roll_events = widening[widening['widening_event'] == True]
print(f"Total rolls detected: {len(roll_events)}")

# Load panel to analyze timing
panel = pd.read_csv('outputs/panels/hg_panel_simple.csv', 
                   index_col=0, low_memory=False)
# Row 1 has expiry dates, remaining rows have prices
```

## 🔬 Methodology

### Core Algorithm: Z-Score Detection

The framework detects rolls through calendar spread widening:

1. **Calendar Spread** = Next Month Price - Front Month Price
2. **Daily Change** = Today's Spread - Yesterday's Spread  
3. **Z-Score** = (Change - Rolling Mean) / Rolling Std Dev
4. **Detection**: Flag if Z-Score > 1.5 (configurable threshold)

### Why This Works
When institutions roll positions:
- They SELL the front month → price pressure down
- They BUY the next month → price pressure up
- Result: Calendar spread WIDENS (detected by our algorithm)

## 📈 Key Findings (Copper Analysis)

| Metric | Value |
|--------|-------|
| **Contracts Analyzed** | 202 (HG copper, 2008-2024) |
| **Trading Days** | 6,206 |
| **Roll Events Detected** | 281 (4.5% of days) |
| **Median Roll Timing** | **12 days before expiry** |
| **Mean Roll Timing** | 13.83 days |
| **Most Common Window** | 5-14 days (40% of rolls) |

### Distribution of Roll Timing
- **0-4 days before expiry**: 18.9% (last-minute rollers)
- **5-9 days**: 21.4% 
- **10-14 days**: 17.4%
- **15-19 days**: 9.6%
- **20-24 days**: 16.7% (early rollers)
- **25-29 days**: 13.2%
- **30+ days**: 2.8%

## 🔄 Analyzing Other Commodities

### Available Commodities (32 types)
```bash
ls organized_data/
# Shows: aluminum, brent_crude, cocoa, coffee, copper, corn, cotton,
#        crude_oil, currencies, equity_indices, feeder_cattle, 
#        gasoline, gold, heating_oil, lean_hogs, live_cattle,
#        natural_gas, oats, orange_juice, palladium, platinum,
#        rice, silver, soybean_meal, soybean_oil, soybeans,
#        sugar, volatility, wheat, and more...
```

### To Analyze Gold (Example)
1. Edit `etf_roll_analysis/config/settings.yaml`:
   ```yaml
   data:
     minute_root: "../organized_data/gold"
   ```

2. Run analysis:
   ```bash
   python scripts/hg_analysis.py  # Works for any commodity
   ```

3. Results appear in same output directories

## 🛠️ Advanced Configuration

### Adjust Detection Sensitivity
In `config/settings.yaml`:
```yaml
spread:
  z_threshold: 1.0   # Lower = more sensitive (more events detected)
  window: 30         # Larger = smoother (fewer false positives)
  cool_down: 5       # Larger = fewer duplicate signals
```

### Use Different Price Fields
```yaml
data:
  price_field: "close"  # Options: "open", "high", "low", "close"
```

## 📚 Documentation

- **[etf_roll_analysis/README.md](etf_roll_analysis/README.md)**: Technical framework details
- **[etf_roll_analysis/USER_GUIDE.md](etf_roll_analysis/USER_GUIDE.md)**: Extensive usage examples
- **[presentation_docs/analysis_report.pdf](presentation_docs/analysis_report.pdf)**: Academic paper (20 pages)

## 🧪 Validation

The analysis has been validated against:
- Official CME expiry dates (100% match rate)
- Volume-based liquidity transitions (65.9% confirmation)
- Statistical significance tests (z-score methodology)

## 👥 Authors

- **Austin Li** - Primary developer
- Analysis framework developed with AI assistance
- Supervisor: Quantitative Finance Professor

## 📄 License

Internal use only - Academic research project for futures roll analysis

## 🤝 Contributing

For questions or improvements:
1. Check existing documentation
2. Review code comments in `src/roll_analysis/`
3. Run validation scripts in `scripts/`

## 🎓 Academic Context

This project was developed to answer the research question: **"When do market participants roll their futures positions?"**

The framework provides empirical evidence that there is no single "roll date" in copper futures markets, but rather heterogeneous strategies with a median timing of 12 days before contract expiration.

---

**For detailed technical documentation, see [etf_roll_analysis/README.md](etf_roll_analysis/README.md)**

**For usage examples and tutorials, see [etf_roll_analysis/USER_GUIDE.md](etf_roll_analysis/USER_GUIDE.md)**
