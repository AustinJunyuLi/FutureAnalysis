from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from futures_roll_analysis.reporting import generate_report
from futures_roll_analysis.results import AnalysisBundle, BucketAnalysisResult, DailyAnalysisResult


def _dummy_bucket() -> BucketAnalysisResult:
    index = pd.date_range("2025-01-01", periods=3, freq="h")
    panel = pd.DataFrame({("HGZ2025", "close"): [4.0, 4.1, 4.2]})
    bucket_summary = pd.DataFrame(
        {
            "bucket": [1, 2],
            "bucket_label": ["US Open", "US Morning"],
            "session": ["US Regular", "US Regular"],
            "event_count": [5, 3],
        }
    )
    bucket_summary["event_share_pct"] = bucket_summary["event_count"] / bucket_summary["event_count"].sum() * 100
    session_summary = bucket_summary.groupby("session", as_index=False)["event_count"].sum()
    session_summary["event_share_pct"] = 100.0
    preference = pd.Series([0.1, 0.2, 0.3], index=index)
    transitions = pd.DataFrame([[0, 1], [1, 0]], columns=[1, 2], index=[1, 2])
    event_summary = pd.DataFrame({"date": index, "change": [0.1, 0.0, 0.2]})
    multi_spreads = pd.DataFrame({"S1": [0.4, 0.5, 0.6]}, index=index)
    multi_events = pd.DataFrame({"S1_events": [True, False, True]}, index=index)
    strip_diag = pd.DataFrame({"front_contract": ["HG"], "classification": ["normal"]})
    metadata = pd.DataFrame({"contract": ["HGZ2025"], "expiry_date": ["2025-12-27"]})
    front_next = pd.DataFrame({"front_contract": ["HGZ2025"] * 3, "next_contract": ["HGZ2026"] * 3}, index=index)
    return BucketAnalysisResult(
        panel=panel,
        front_next=front_next,
        spread=pd.Series([0.1, 0.2, 0.3], index=index),
        widening=pd.Series([True, False, True], index=index),
        bucket_summary=bucket_summary,
        session_summary=session_summary,
        preference_scores=preference,
        transition_matrix=transitions,
        event_summary=event_summary,
        business_days_index=pd.DatetimeIndex([]),
        business_days_audit=None,
        multi_spreads=multi_spreads,
        multi_events=multi_events,
        strip_diagnostics=strip_diag,
        metadata=metadata,
    )


def _dummy_daily() -> DailyAnalysisResult:
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    panel = pd.DataFrame({("HGZ2025", "close"): [4.0, 4.1, 4.2, 4.3]}, index=index)
    front_next = pd.DataFrame({"front_contract": ["HGZ2025"] * 4, "next_contract": ["HGZ2026"] * 4}, index=index)
    summary = pd.DataFrame(
        {
            "date": index,
            "change": [0.1, 0.2, 0.0, 0.3],
            "business_days_since_last": [None, 3, 1, 2],
        }
    )
    quality = pd.DataFrame(
        {
            "contract": ["HGH2025", "HGK2025"],
            "status": ["INCLUDED", "EXCLUDED"],
        }
    )
    metadata = pd.DataFrame({"contract": ["HGZ2025"], "expiry_date": ["2025-12-27"]})
    return DailyAnalysisResult(
        panel=panel,
        front_next=front_next,
        spread=pd.Series([0.3, 0.4, 0.5, 0.6], index=index),
        widening=pd.Series([False, True, False, True], index=index),
        liquidity=pd.Series([0.7, 0.9, 0.85, 1.1], index=index),
        summary=summary,
        quality_metrics=quality,
        business_days_index=pd.DatetimeIndex([]),
        business_days_audit=None,
        metadata=metadata,
    )


def test_generate_report_creates_sections(tmp_path):
    bundle = AnalysisBundle(bucket=_dummy_bucket(), daily=_dummy_daily())
    dummy_settings = SimpleNamespace(products=["HG"])
    out_path = tmp_path / "report.tex"
    generate_report(bundle, dummy_settings, out_path)

    contents = out_path.read_text(encoding="utf-8")
    assert "\\section{Executive Summary}" in contents
    assert "\\section{Methodology and Data Sources}" in contents
    assert "\\section{Data Coverage}" in contents
    assert "\\section{Hourly Bucket Analysis}" in contents
    assert "\\section{Multi-Spread Diagnostics}" in contents
    assert "\\section{Timing Relative to Expiry}" in contents
    assert "\\section{Daily Aggregation}" in contents
    assert contents.count("\\begin{tikzpicture}") >= 3
