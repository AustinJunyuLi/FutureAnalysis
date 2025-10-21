from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    filter_enabled: bool = True
    cutoff_year: int = 2015
    min_data_points: int = 1000
    max_gap_days: int = 30
    min_coverage_percent: float = 30.0
    trim_early_sparse: bool = True
    commodity: str = "HG"


class DataQualityFilter:
    """Evaluate and optionally filter contracts based on configurable quality criteria."""

    def __init__(self, config: Dict[str, object]):
        self.config = QualityConfig(**config)

    def apply(
        self, daily_by_contract: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
        if not self.config.filter_enabled:
            return daily_by_contract, []

        filtered: Dict[str, pd.DataFrame] = {}
        metrics: List[dict] = []

        for contract, frame in daily_by_contract.items():
            if self.config.commodity and not contract.startswith(self.config.commodity):
                filtered[contract] = frame
                continue

            summary = self._evaluate_contract(frame, contract)
            metrics.append(summary)

            if summary["status"] == "INCLUDED":
                trimmed = self._trim(frame) if self.config.trim_early_sparse else frame
                filtered[contract] = trimmed

        included = sum(1 for metric in metrics if metric["status"] == "INCLUDED")
        excluded = len(metrics) - included
        LOGGER.info(
            "Data quality filter: total=%s included=%s excluded=%s",
            len(metrics),
            included,
            excluded,
        )
        return filtered, metrics

    def _evaluate_contract(self, df: pd.DataFrame, contract: str) -> dict:
        if df.empty:
            return {
                "contract": contract,
                "status": "EXCLUDED",
                "reasons": ["Empty dataset"],
                "data_points": 0,
            }

        data_points = len(df)
        date_range = (df.index.min(), df.index.max())
        total_days = (date_range[1] - date_range[0]).days + 1

        date_diffs = df.index.to_series().diff()
        gap_indices = np.where(date_diffs > pd.Timedelta(days=5))[0]
        gaps = []
        for idx in gap_indices:
            start = df.index[idx - 1]
            end = df.index[idx]
            gap_days = (end - start).days
            if gap_days > self.config.max_gap_days:
                gaps.append(
                    {
                        "start": start.strftime("%Y-%m-%d"),
                        "end": end.strftime("%Y-%m-%d"),
                        "days": gap_days,
                    }
                )

        expected_trading_days = total_days * 0.7
        coverage = (
            data_points / expected_trading_days * 100 if expected_trading_days > 0 else 0
        )

        year = _extract_year(contract)
        reasons = []
        status = "INCLUDED"

        if year and year < self.config.cutoff_year:
            reasons.append(f"Contract year {year} before cutoff {self.config.cutoff_year}")
            status = "EXCLUDED"
        if data_points < self.config.min_data_points:
            reasons.append(f"Only {data_points} data points (min {self.config.min_data_points})")
            status = "EXCLUDED"
        if gaps:
            max_gap = max(gap["days"] for gap in gaps)
            reasons.append(f"Large gaps in data (max {max_gap} days)")
            status = "EXCLUDED"
        if coverage < self.config.min_coverage_percent:
            reasons.append(
                f"Coverage {coverage:.1f}% (min {self.config.min_coverage_percent}%)"
            )
            status = "EXCLUDED"

        return {
            "contract": contract,
            "status": status,
            "reasons": reasons,
            "data_points": data_points,
            "date_range": f"{date_range[0].date()} to {date_range[1].date()}",
            "coverage_percent": round(coverage, 1),
            "gaps": gaps,
            "max_gap_days": max((gap["days"] for gap in gaps), default=0),
            "year": year,
        }

    def _trim(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 20:
            return df
        window = 20
        for start in range(len(df) - window):
            window_slice = df.iloc[start : start + window]
            max_gap = window_slice.index.to_series().diff().max()
            if max_gap <= pd.Timedelta(days=5):
                return df[df.index >= window_slice.index[0]]
        return df

    @staticmethod
    def save_reports(metrics: List[dict], output_dir: Path, commodity: str) -> None:
        if not metrics:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame(metrics)
        metrics_csv = output_dir / f"{commodity}_quality_metrics.csv"
        metrics_csv.unlink(missing_ok=True)
        metrics_df.to_csv(metrics_csv, index=False)
        summary = {
            "commodity": commodity,
            "total_contracts": len(metrics),
            "excluded": int((metrics_df["status"] == "EXCLUDED").sum()),
            "included": int((metrics_df["status"] == "INCLUDED").sum()),
        }
        summary_path = output_dir / f"{commodity}_filtering_summary.json"
        summary_path.unlink(missing_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
def _extract_year(contract: str) -> int:
    match = re.search(r"[A-Z]+(\d{2,4})$", contract)
    if not match:
        return 0
    year_str = match.group(1)
    year = int(year_str)
    if len(year_str) == 2:
        year = 2000 + year if year <= 30 else 1900 + year
    return year
