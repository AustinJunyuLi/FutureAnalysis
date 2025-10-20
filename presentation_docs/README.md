# Analysis Report - Futures Roll Analysis

This directory contains the formal analysis report for the copper futures roll analysis project.

## Document

### `analysis_report.tex`
**Audience**: Technical/academic (quantitative finance)

**Content**:
- Formal academic structure
- Mathematical notation and statistical methodology
- Detailed results with tables
- Technical implementation details
- 15-20 pages

**Key Sections**:
- Executive Summary
- Methodology (z-score detection)
- Results (281 events, 12-day median)
- Discussion and interpretation
- Future work

**Note**: Personal reference materials have been moved to Desktop folder

## Compilation

### Requirements
- LaTeX distribution (TeXLive, MiKTeX, or MacTeX)
- pdflatex compiler

### Quick Start

#### Option 1: Using Make (Linux/WSL/Mac)
```bash
# Compile analysis report
make

# or explicitly
make analysis

# Clean auxiliary files
make clean

# View help
make help
```

#### Option 2: Using pdflatex directly

**Windows (PowerShell/CMD)**:
```cmd
pdflatex analysis_report.tex
pdflatex analysis_report.tex
```

**Linux/Mac/WSL**:
```bash
pdflatex analysis_report.tex
pdflatex analysis_report.tex
```

Note: Run pdflatex twice to resolve cross-references and table of contents.

### Compilation Issues

**"Command not found: pdflatex"**
- Install LaTeX: 
  - Ubuntu/Debian: `sudo apt-get install texlive-full`
  - Mac: `brew install --cask mactex`
  - Windows: Download MiKTeX from https://miktex.org/

**Missing packages**
- Most distributions auto-install missing packages
- Or install full distribution: `texlive-full` (Linux) or complete MiKTeX/MacTeX

## Generated Files

After compilation:
```
presentation_docs/
├── analysis_report.pdf        # Technical report
├── analysis_report.tex
├── Makefile
├── README.md (this file)
└── (auxiliary files: .aux, .log, .out, .toc)
```

## Quick Reference

### Key Findings to Remember

**Dataset**:
- 202 HG copper contracts
- 6,206 trading days (2008-2024)
- 1-minute data aggregated to daily

**Main Result**:
- **Median roll timing: 12 days before expiry**
- Mean: 13.83 days
- 281 events detected

**Distribution**:
- 40% roll in 5-14 day window
- 19% roll in final 0-4 days
- Multiple strategies (not one "roll date")

**Method**:
- Z-score detection (window=20, threshold=1.5)
- Calendar spread widening = rolling signal
- 100% validation against CME expiry dates

## Presentation Tips

1. **Opening (30 sec)**: "I analyzed 16 years of copper futures to detect roll timing. Found median of 12 days before expiry with heterogeneous strategies."

2. **Key Numbers**: 281 events, 12 days, 40% in 5-14 window

3. **Technical Depth**: Adapt to questions - can go from layman (spreads widen) to technical (z-score methodology)

4. **Confidence Builders**:
   - You organized 13,548 files
   - Built complete statistical framework
   - Found economically meaningful results
   - Framework ready for 31 other commodities

## File Structure

```
../                                  # Project root
├── organized_data/                  # Organized futures data
│   ├── copper/                      # 202 HG contract files
│   ├── gold/
│   └── (30 other commodities)
├── etf_roll_analysis/               # Analysis framework
│   ├── src/roll_analysis/           # Core modules
│   │   ├── ingest.py
│   │   ├── panel.py
│   │   ├── rolls.py
│   │   ├── spread.py
│   │   └── events.py
│   ├── scripts/
│   │   └── hg_analysis.py           # Main pipeline
│   ├── config/
│   │   └── settings.yaml
│   └── outputs/                     # Analysis results
│       ├── panels/
│       │   ├── hg_panel_simple.csv
│       │   └── hg_panel.parquet
│       └── roll_signals/
│           ├── hg_widening.csv      # 281 events
│           ├── hg_spread.csv
│           └── hg_liquidity_roll.csv
└── presentation_docs/               # THIS DIRECTORY
    ├── analysis_report.tex
    ├── analysis_report.pdf
    ├── Makefile
    └── README.md
```

**Personal Reference Materials**:
These have been moved to:
```
C:\Users\Austin Li\Desktop\Personal_Reference_Materials\
```

Contains:
- KEY_POINTS.txt - Quick presentation guide
- personal_reference.pdf - Complete study guide
- Compilation instructions
- Q&A preparation

## Support

For LaTeX compilation issues:
- Check pdflatex is installed: `pdflatex --version`
- Ensure all .tex files are in same directory
- Run pdflatex twice for cross-references

For content questions:
- See Personal_Reference_Materials on Desktop
- Review KEY_POINTS.txt for presentation
- Check analysis outputs in `../etf_roll_analysis/outputs/`

## License

These documents are for internal use and presentation purposes related to the futures roll analysis project.

---

**Good luck with your presentation!**
