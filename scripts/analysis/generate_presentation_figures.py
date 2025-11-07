#!/usr/bin/env python3
"""
Generate presentation figures for analysis report.

Regenerates bucket_distribution_bar.png and preference_scores_bar.png
from current analysis outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Paths
project_root = Path(__file__).parent.parent.parent  # Go up to repo root
outputs_dir = project_root / "outputs" / "latest_hourly" / "analysis"
figures_dir = project_root / "presentation_docs" / "figures"

# Ensure figures directory exists
figures_dir.mkdir(parents=True, exist_ok=True)

print("Generating presentation figures...")

# Figure 1: Bucket Distribution
print("  1. bucket_distribution_bar.png")
try:
    bucket_summary = pd.read_csv(outputs_dir / "bucket_summary.csv")

    plt.figure(figsize=(10, 6))
    plt.bar(bucket_summary['bucket'], bucket_summary['event_count'], color='steelblue', alpha=0.7)
    plt.xlabel('Bucket (Hourly Period)', fontsize=12, fontweight='bold')
    plt.ylabel('Event Count', fontsize=12, fontweight='bold')
    plt.title('Spread Widening Events by Intraday Period', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "bucket_distribution_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✓ Generated")
except Exception as e:
    print(f"     ✗ Error: {e}")

# Figure 2: Preference Scores
print("  2. preference_scores_bar.png")
try:
    preference_scores = pd.read_csv(outputs_dir / "preference_scores.csv")
    # Get the actual column name (might be '0' or 'preference_score')
    score_col = preference_scores.columns[1]

    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if x > 1.5 else '#2ca02c' if x < 0.5 else '#1f77b4'
              for x in preference_scores[score_col]]
    plt.bar(preference_scores['bucket'], preference_scores[score_col],
            color=colors, alpha=0.7)
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5,
                label='Neutral (1.0)')
    plt.xlabel('Bucket (Hourly Period)', fontsize=12, fontweight='bold')
    plt.ylabel('Preference Score', fontsize=12, fontweight='bold')
    plt.title('Event Concentration by Period (Preference Score)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "preference_scores_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✓ Generated")
except Exception as e:
    print(f"     ✗ Error: {e}")

print("\nFigure generation complete!")
print(f"Outputs saved to: {figures_dir}")
