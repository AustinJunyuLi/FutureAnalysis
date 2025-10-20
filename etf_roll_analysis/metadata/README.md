# Expiry metadata (HG)

This folder stores explicit expiry dates for the contracts you analyze. For HG (COMEX Copper), populate `contracts_metadata.csv` with:

Required columns:
- root: HG
- contract: e.g., HGZ2025 (root + month code + 4-digit year)
- expiry_date: official last trade/termination date (YYYY-MM-DD)
- source: e.g., "CME product calendar"
- source_url: URL to the specific page you used (ideally a link that documents the date)

Recommended source for HG:
- CME Group Copper Product Calendar:
  - https://www.cmegroup.com/markets/metals/base/copper.calendar.html
- CME Group Copper Contract Specs (for the termination rule):
  - https://www.cmegroup.com/markets/metals/base/copper.contractSpecs.html

Workflow
1) Discover which HG contracts you have data for by running the script once; it will list missing contracts if metadata is incomplete.
2) For each missing contract, look up the official expiry date on CME and add a row to `contracts_metadata.csv` with the date, source, and source_url.
3) Re-run the script. It will now proceed and record the source columns in the output tables.

Notes
- The analysis will not proceed without explicit expiry metadata for all discovered contracts.
- Ensure dates are in exchange-local terms. If the source indicates a time cutoff, use the calendar date of termination.
