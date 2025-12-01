#!/usr/bin/env python3
"""
Volume Migration & Crossover Analysis for Futures Calendar Spreads

This script analyzes when F2 (next) contract volume surpasses F1 (front) contract volume
and correlates this migration with spread widening events.

IMPORTANT CAVEAT:
================
Volume measures trading activity (contracts traded), not position holdings (open interest).
High volume may reflect day trading rather than position migration. This analysis uses volume
as an IMPERFECT PROXY for liquidity shifts, recognizing it cannot directly measure OI migration
patterns. The supervisor's observation about OI migration would be more directly tested with
actual open interest data.

Research Question:
When does F2 volume typically surpass F1 volume relative to F1 expiry, and does this correlate
with spread widening events?
"""

from __future__ import annotations

# Fix matplotlib backend and fonts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'sans-serif']

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_f1_f2_volumes(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    expiry_map: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract F1 and F2 volumes from the panel using front_contract and next_contract metadata.

    Parameters
    ----------
    panel : pd.DataFrame
        Hourly panel with MultiIndex columns (contract, field)
    metadata : pd.DataFrame
        Contract metadata with expiry dates
    expiry_map : pd.Series
        Series mapping contract codes to expiry dates

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (f1_volumes_df, f2_volumes_df) - DataFrames with F1 and F2 volumes indexed by timestamp
    """
    logger.info("Extracting F1 and F2 volumes from panel...")

    # Get all contracts from panel
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != "meta"
    ]
    logger.info(f"Found {len(contracts)} contracts in panel")

    # Extract volume field for all contracts
    volume_tuples = [
        (contract, "volume") for contract in contracts if (contract, "volume") in panel.columns
    ]
    if not volume_tuples:
        raise ValueError("Panel does not contain volume data")

    volumes = panel.loc[:, volume_tuples].copy()
    volumes.columns = [contract for contract, _ in volumes.columns]

    # Get F1 and F2 contract names from metadata
    f1_contracts = panel[("meta", "front_contract")].values
    f2_contracts = panel[("meta", "next_contract")].values

    # Create mapping from contract name to column index
    contract_index = {contract: idx for idx, contract in enumerate(volumes.columns)}

    # Extract F1 and F2 volumes
    f1_volumes = np.full(len(panel), np.nan)
    f2_volumes = np.full(len(panel), np.nan)

    for i, (f1_contract, f2_contract) in enumerate(zip(f1_contracts, f2_contracts)):
        if f1_contract and f1_contract in contract_index:
            col_idx = contract_index[f1_contract]
            f1_volumes[i] = volumes.iloc[i, col_idx]
        if f2_contract and f2_contract in contract_index:
            col_idx = contract_index[f2_contract]
            f2_volumes[i] = volumes.iloc[i, col_idx]

    f1_vol_series = pd.Series(f1_volumes, index=panel.index, name="f1_volume")
    f2_vol_series = pd.Series(f2_volumes, index=panel.index, name="f2_volume")

    logger.info(f"F1 volume: {f1_vol_series.notna().sum()} non-null values")
    logger.info(f"F2 volume: {f2_vol_series.notna().sum()} non-null values")

    return f1_vol_series, f2_vol_series


def calculate_volume_ratio(
    f1_volumes: pd.Series,
    f2_volumes: pd.Series,
    min_volume: float = 1.0,
) -> pd.Series:
    """
    Calculate F2/F1 volume ratio.

    Parameters
    ----------
    f1_volumes : pd.Series
        F1 (front) contract volumes
    f2_volumes : pd.Series
        F2 (next) contract volumes
    min_volume : float
        Minimum F1 volume for ratio to be valid (avoids division near zero)

    Returns
    -------
    pd.Series
        F2/F1 volume ratio (NaN where F1 volume < min_volume or missing)
    """
    logger.info("Calculating F2/F1 volume ratio...")
    ratio = pd.Series(index=f1_volumes.index, dtype=float)
    valid_mask = (f1_volumes >= min_volume) & (f2_volumes.notna()) & (f1_volumes.notna())
    ratio[valid_mask] = f2_volumes[valid_mask] / f1_volumes[valid_mask]
    logger.info(f"Volume ratio: {valid_mask.sum()} valid ratios")
    return ratio


def identify_crossover_events(
    volume_ratio: pd.Series,
    f1_contracts: pd.Series,
    expiry_map: pd.Series,
) -> pd.DataFrame:
    """
    Identify when F2 volume first exceeds F1 volume (ratio > 1.0) within each contract cycle.

    Parameters
    ----------
    volume_ratio : pd.Series
        F2/F1 volume ratio
    f1_contracts : pd.Series
        F1 contract names at each timestamp
    expiry_map : pd.Series
        Contract to expiry date mapping

    Returns
    -------
    pd.DataFrame
        Crossover events with columns: timestamp, f1_contract, crossover_ratio,
        days_to_f1_expiry, trading_day
    """
    logger.info("Identifying volume crossover events...")

    crossovers = []
    current_f1 = None
    crossed_over = False

    for ts, ratio, f1 in zip(volume_ratio.index, volume_ratio.values, f1_contracts.values):
        # Reset when F1 contract changes
        if f1 != current_f1:
            current_f1 = f1
            crossed_over = False

        # Detect first crossover for this contract pair
        if not crossed_over and pd.notna(ratio) and ratio > 1.0:
            if f1 in expiry_map.index:
                expiry_date = expiry_map[f1]
                days_to_expiry = (expiry_date - ts.normalize()).days
                crossovers.append(
                    {
                        "timestamp": ts,
                        "f1_contract": f1,
                        "crossover_ratio": ratio,
                        "days_to_f1_expiry": days_to_expiry,
                        "trading_day": ts.normalize(),
                    }
                )
                crossed_over = True
                logger.debug(
                    f"Crossover: {f1} on {ts} (ratio={ratio:.2f}, "
                    f"days_to_expiry={days_to_expiry})"
                )

    result = pd.DataFrame(crossovers)
    if result.empty:
        logger.warning("No crossover events found!")
        return result

    logger.info(f"Found {len(result)} crossover events")
    logger.info(
        f"Median days to expiry at crossover: {result['days_to_f1_expiry'].median():.1f}"
    )
    return result


def analyze_crossover_timing(crossovers: pd.DataFrame) -> dict:
    """
    Analyze timing statistics of crossover events.

    Parameters
    ----------
    crossovers : pd.DataFrame
        Crossover events from identify_crossover_events()

    Returns
    -------
    dict
        Summary statistics: median, mean, std, min, max, IQR
    """
    if crossovers.empty:
        logger.warning("No crossover data to analyze")
        return {}

    timing = crossovers["days_to_f1_expiry"].dropna()

    stats_dict = {
        "count": len(timing),
        "median": timing.median(),
        "mean": timing.mean(),
        "std": timing.std(),
        "min": timing.min(),
        "max": timing.max(),
        "q25": timing.quantile(0.25),
        "q75": timing.quantile(0.75),
        "iqr": timing.quantile(0.75) - timing.quantile(0.25),
    }

    logger.info("\n=== CROSSOVER TIMING STATISTICS ===")
    logger.info(f"Total crossovers: {stats_dict['count']}")
    logger.info(f"Median: {stats_dict['median']:.1f} days before F1 expiry")
    logger.info(f"Mean: {stats_dict['mean']:.1f} days (std={stats_dict['std']:.1f})")
    logger.info(f"Range: {stats_dict['min']:.0f} to {stats_dict['max']:.0f} days")
    logger.info(f"IQR: {stats_dict['q25']:.1f} to {stats_dict['q75']:.1f} days")

    # Test supervisor's hypothesis: 14-day window?
    in_14_day = (timing >= 7) & (timing <= 21)
    pct_14_day = 100 * in_14_day.sum() / len(timing)
    logger.info(f"% in 7-21 day window: {pct_14_day:.1f}%")

    in_14_day_exact = (timing >= 10) & (timing <= 18)
    pct_14_day_exact = 100 * in_14_day_exact.sum() / len(timing)
    logger.info(f"% in 10-18 day window: {pct_14_day_exact:.1f}%")

    return stats_dict


def correlate_with_spread_events(
    volume_ratio: pd.Series,
    widening_events: pd.Series,
) -> dict:
    """
    Correlate volume ratio with spread widening events.

    Parameters
    ----------
    volume_ratio : pd.Series
        F2/F1 volume ratio
    widening_events : pd.Series
        Boolean series of spread widening events

    Returns
    -------
    dict
        Correlation statistics
    """
    logger.info("\n=== VOLUME-EVENT CORRELATION ANALYSIS ===")

    # Filter to valid data
    valid = volume_ratio.notna() & widening_events.notna()
    ratio_clean = volume_ratio[valid]
    events_clean = widening_events[valid]

    logger.info(f"Valid ratio-event pairs: {valid.sum()} / {len(valid)}")

    if valid.sum() < 10:
        logger.warning("Not enough valid pairs for correlation analysis")
        return {}

    # Pearson correlation
    corr, p_value = stats.pearsonr(ratio_clean, events_clean.astype(int))
    logger.info(f"Pearson correlation: {corr:.4f} (p={p_value:.4e})")

    # Compare ratio distribution: event vs non-event periods
    ratio_during_events = ratio_clean[events_clean == 1]
    ratio_no_events = ratio_clean[events_clean == 0]

    logger.info(f"\nRatio during widening events:")
    logger.info(f"  Count: {len(ratio_during_events)}")
    logger.info(f"  Median: {ratio_during_events.median():.3f}")
    logger.info(f"  Mean: {ratio_during_events.mean():.3f}")

    logger.info(f"\nRatio during non-widening periods:")
    logger.info(f"  Count: {len(ratio_no_events)}")
    logger.info(f"  Median: {ratio_no_events.median():.3f}")
    logger.info(f"  Mean: {ratio_no_events.mean():.3f}")

    # T-test
    if len(ratio_during_events) > 0 and len(ratio_no_events) > 0:
        t_stat, t_pval = stats.ttest_ind(
            ratio_during_events.dropna(), ratio_no_events.dropna(), equal_var=False
        )
        logger.info(f"\nWelch t-test: t={t_stat:.4f}, p={t_pval:.4e}")

    # Event likelihood at different ratio thresholds
    for threshold in [0.8, 1.0, 1.5, 2.0]:
        high_ratio = ratio_clean >= threshold
        if high_ratio.sum() > 0:
            event_rate = events_clean[high_ratio].mean()
            total_rate = events_clean.mean()
            lift = event_rate / total_rate if total_rate > 0 else 0
            logger.info(
                f"Event rate when ratio >= {threshold}: "
                f"{100*event_rate:.2f}% (lift={lift:.2f}x baseline)"
            )

    return {
        "pearson_r": corr,
        "pearson_p": p_value,
        "ratio_median_during_events": ratio_during_events.median(),
        "ratio_median_no_events": ratio_no_events.median(),
    }


def create_timeseries_visualization(
    volume_ratio: pd.Series,
    widening_events: pd.Series,
    f1_contracts: pd.Series,
    crossovers: pd.DataFrame,
    expiry_map: pd.Series,
    output_path: Path,
) -> None:
    """
    Create timeseries visualization of volume ratio and spread widening events.

    Parameters
    ----------
    volume_ratio : pd.Series
        F2/F1 volume ratio
    widening_events : pd.Series
        Boolean spread widening events
    f1_contracts : pd.Series
        F1 contract names
    crossovers : pd.DataFrame
        Identified crossover events
    expiry_map : pd.Series
        Contract expiry dates
    output_path : Path
        Where to save PNG
    """
    logger.info(f"Creating timeseries visualization...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top panel: Volume ratio
    ax1.plot(volume_ratio.index, volume_ratio.values, linewidth=1, color="steelblue", alpha=0.7)
    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Crossover threshold (ratio=1.0)")
    ax1.fill_between(
        volume_ratio.index, 0, 1, where=(volume_ratio >= 1.0), alpha=0.2, color="green", label="F2 > F1"
    )
    ax1.set_ylabel("F2/F1 Volume Ratio", fontsize=12, fontweight="bold")
    ax1.set_title("Volume Migration from F1 to F2 and Spread Widening Events", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.set_ylim([0, 3])

    # Mark crossover events
    if not crossovers.empty:
        ax1.scatter(
            crossovers["timestamp"],
            crossovers["crossover_ratio"],
            color="gold",
            s=100,
            marker="*",
            zorder=5,
            label=f"Crossover events (n={len(crossovers)})",
        )
        ax1.legend(loc="upper left")

    # Bottom panel: Spread widening events
    event_timestamps = widening_events[widening_events == 1].index
    ax2.scatter(event_timestamps, np.ones(len(event_timestamps)), color="red", alpha=0.5, s=20, label="Widening events")
    ax2.set_ylabel("Spread Widening", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, 1.5])
    ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")
    plt.close()


def main(
    panel_path: Path = Path("outputs/panels/hourly_panel.parquet"),
    metadata_path: Path = Path("metadata/contracts_metadata.csv"),
    widening_path: Path = Path("outputs/roll_signals/hourly_widening.csv"),
    output_dir: Path = Path("outputs/exploratory"),
) -> None:
    """
    Main analysis pipeline.

    Parameters
    ----------
    panel_path : Path
        Path to hourly panel parquet file
    metadata_path : Path
        Path to contracts metadata CSV
    widening_path : Path
        Path to spread widening events CSV
    output_dir : Path
        Output directory for results
    """
    logger.info("=" * 70)
    logger.info("VOLUME MIGRATION & CROSSOVER ANALYSIS")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\nLoading data...")
    panel = pd.read_parquet(panel_path)
    metadata = pd.read_csv(metadata_path)
    widening = pd.read_csv(widening_path, index_col=0, parse_dates=True)

    # Build expiry map
    expiry_map = (
        metadata.drop_duplicates("contract")
        .set_index("contract")["expiry_date"]
        .pipe(pd.to_datetime)
        .dt.normalize()
    )

    # Extract volumes
    f1_volumes, f2_volumes = extract_f1_f2_volumes(panel, metadata, expiry_map)

    # Calculate ratio
    volume_ratio = calculate_volume_ratio(f1_volumes, f2_volumes)

    # Create output dataframe with all metrics
    combined = pd.DataFrame(
        {
            "f1_volume": f1_volumes,
            "f2_volume": f2_volumes,
            "volume_ratio": volume_ratio,
            "f1_contract": panel[("meta", "front_contract")],
            "f2_contract": panel[("meta", "next_contract")],
            "widening_event": widening.iloc[:, 0].reindex(panel.index),
        }
    )

    # Identify crossovers
    crossovers = identify_crossover_events(volume_ratio, panel[("meta", "front_contract")], expiry_map)

    # Analyze timing
    timing_stats = analyze_crossover_timing(crossovers)

    # Correlate with events
    correlation_stats = correlate_with_spread_events(volume_ratio, combined["widening_event"])

    # Create visualization
    viz_path = output_dir / "volume_ratio_timeseries.png"
    create_timeseries_visualization(
        volume_ratio, combined["widening_event"], combined["f1_contract"], crossovers, expiry_map, viz_path
    )

    # Write outputs
    logger.info("\nWriting outputs...")

    # 1. Full combined timeseries
    combined.to_csv(output_dir / "volume_ratio_timeseries.csv")
    logger.info(f"Wrote: {output_dir / 'volume_ratio_timeseries.csv'}")

    # 2. Crossover events
    if not crossovers.empty:
        crossovers.to_csv(output_dir / "volume_crossovers.csv", index=False)
        logger.info(f"Wrote: {output_dir / 'volume_crossovers.csv'}")

    # 3. Summary statistics
    summary_stats = pd.DataFrame(
        {
            "metric": list(timing_stats.keys()),
            "value": list(timing_stats.values()),
        }
    )
    summary_stats.to_csv(output_dir / "crossover_timing_summary.csv", index=False)
    logger.info(f"Wrote: {output_dir / 'crossover_timing_summary.csv'}")

    # 4. Correlation summary
    correlation_df = pd.DataFrame(
        {
            "metric": ["pearson_r", "pearson_p", "ratio_median_during_events", "ratio_median_no_events"],
            "value": [
                correlation_stats.get("pearson_r"),
                correlation_stats.get("pearson_p"),
                correlation_stats.get("ratio_median_during_events"),
                correlation_stats.get("ratio_median_no_events"),
            ],
        }
    )
    correlation_df.to_csv(output_dir / "volume_event_correlation.csv", index=False)
    logger.info(f"Wrote: {output_dir / 'volume_event_correlation.csv'}")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info("\nKey Findings:")
    logger.info(f"  - {len(crossovers)} crossover events identified")
    if timing_stats:
        logger.info(f"  - Median timing: {timing_stats.get('median', 0):.1f} days before F1 expiry")
    logger.info(f"  - Correlation with spread widening: {correlation_stats.get('pearson_r', 0):.4f}")
    logger.info("\nIMPORTANT CAVEAT:")
    logger.info("  Volume ≠ Open Interest. This analysis uses volume as a liquidity proxy,")
    logger.info("  recognizing that volume cannot directly measure OI migration patterns.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
