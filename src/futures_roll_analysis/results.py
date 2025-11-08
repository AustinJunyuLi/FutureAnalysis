from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class BucketAnalysisResult:
    panel: pd.DataFrame
    front_next: pd.DataFrame
    spread: pd.Series
    widening: pd.Series
    bucket_summary: pd.DataFrame
    session_summary: pd.DataFrame
    preference_scores: pd.Series
    transition_matrix: pd.DataFrame
    event_summary: pd.DataFrame
    business_days_index: Optional[pd.DatetimeIndex]
    business_days_audit: Optional[pd.DataFrame]
    multi_spreads: pd.DataFrame
    multi_events: pd.DataFrame
    strip_diagnostics: pd.DataFrame
    metadata: pd.DataFrame


@dataclass(frozen=True)
class DailyAnalysisResult:
    panel: pd.DataFrame
    front_next: pd.DataFrame
    spread: pd.Series
    widening: pd.Series
    liquidity: pd.Series
    summary: pd.DataFrame
    quality_metrics: pd.DataFrame
    business_days_index: Optional[pd.DatetimeIndex]
    business_days_audit: Optional[pd.DataFrame]
    metadata: pd.DataFrame


@dataclass(frozen=True)
class AnalysisBundle:
    bucket: Optional[BucketAnalysisResult] = None
    daily: Optional[DailyAnalysisResult] = None
