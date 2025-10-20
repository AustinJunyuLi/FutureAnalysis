"""
Data quality filtering module for futures contract analysis.
Identifies and filters problematic contracts based on data quality metrics.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class DataQualityFilter:
    """Filter futures contracts based on data quality criteria."""
    
    def __init__(self, config: dict):
        """
        Initialize filter with configuration parameters.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with filtering parameters
        """
        self.filter_enabled = config.get('filter_enabled', True)
        self.cutoff_year = config.get('cutoff_year', 2015)
        self.min_data_points = config.get('min_data_points', 1000)
        self.max_gap_days = config.get('max_gap_days', 30)
        self.min_coverage_percent = config.get('min_coverage_percent', 30)
        self.trim_early_sparse = config.get('trim_early_sparse', True)
        self.commodity = config.get('commodity', 'HG')
        
    def analyze_contract_quality(self, df: pd.DataFrame, contract_name: str) -> dict:
        """
        Analyze data quality metrics for a single contract.
        
        Returns dict with quality metrics and identified issues.
        """
        if df.empty:
            return {
                'contract': contract_name,
                'status': 'EXCLUDED',
                'reason': 'Empty dataset',
                'data_points': 0,
                'date_range': None,
                'gaps': []
            }
        
        # Basic metrics
        data_points = len(df)
        date_range = (df.index.min(), df.index.max())
        total_days = (date_range[1] - date_range[0]).days + 1
        
        # Find gaps in data
        gaps = []
        if len(df) > 1:
            date_diffs = pd.Series(df.index).diff()
            gap_mask = date_diffs > pd.Timedelta(days=5)  # Gaps > 5 days
            
            if gap_mask.any():
                gap_indices = np.where(gap_mask)[0]
                for idx in gap_indices:
                    gap_start = df.index[idx - 1]
                    gap_end = df.index[idx]
                    gap_days = (gap_end - gap_start).days
                    if gap_days > self.max_gap_days:
                        gaps.append({
                            'start': gap_start.strftime('%Y-%m-%d'),
                            'end': gap_end.strftime('%Y-%m-%d'),
                            'days': gap_days
                        })
        
        # Calculate coverage
        expected_trading_days = total_days * 0.7  # Roughly 70% of calendar days are trading days
        coverage_percent = (data_points / expected_trading_days * 100) if expected_trading_days > 0 else 0
        
        # Extract year from contract name (e.g., HGF2015 -> 2015, HGF15 -> 2015)
        try:
            # Match pattern like HGF2015 or HGF15
            import re
            match = re.search(r'HG[A-Z](\d{2,4})', contract_name)
            if match:
                year_str = match.group(1)
                if len(year_str) == 4:
                    year = int(year_str)
                else:
                    year = int(year_str)
                    if year <= 30:  # Assume 20XX for years <= 30
                        year = 2000 + year
                    else:
                        year = 1900 + year
            else:
                year = 0
        except:
            year = 0
        
        # Determine status and reasons
        reasons = []
        status = 'INCLUDED'
        
        if year > 0 and year < self.cutoff_year:
            reasons.append(f'Contract year {year} before cutoff {self.cutoff_year}')
            status = 'EXCLUDED'
        
        if data_points < self.min_data_points:
            reasons.append(f'Only {data_points} data points (min: {self.min_data_points})')
            status = 'EXCLUDED'
        
        if gaps:
            max_gap = max(g['days'] for g in gaps)
            reasons.append(f'Large gaps in data (max: {max_gap} days)')
            status = 'EXCLUDED'
        
        if coverage_percent < self.min_coverage_percent:
            reasons.append(f'Low coverage: {coverage_percent:.1f}% (min: {self.min_coverage_percent}%)')
            status = 'EXCLUDED'
        
        return {
            'contract': contract_name,
            'status': status,
            'reasons': reasons,
            'data_points': data_points,
            'date_range': f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}",
            'total_days': total_days,
            'coverage_percent': round(coverage_percent, 1),
            'gaps': gaps,
            'max_gap_days': max(g['days'] for g in gaps) if gaps else 0,
            'year': year
        }
    
    def trim_early_sparse_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove early sparse data from a contract, keeping only the continuous trading period.
        """
        if len(df) < 20:  # Too few points to trim
            return df
        
        # Find the first continuous trading period (no gaps > 5 days for at least 20 days)
        continuous_start = None
        window_size = 20
        
        for i in range(len(df) - window_size):
            window = df.iloc[i:i+window_size]
            date_diffs = pd.Series(window.index).diff()
            max_gap = date_diffs.max()
            
            if max_gap <= pd.Timedelta(days=5):
                continuous_start = window.index[0]
                break
        
        if continuous_start:
            return df[df.index >= continuous_start]
        else:
            return df  # No continuous period found, return as-is
    
    def apply(self, daily_by_contract: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
        """
        Apply filtering to all contracts.
        
        Returns:
        --------
        filtered_contracts : dict
            Contracts that passed quality checks (potentially trimmed)
        excluded_info : list
            Information about excluded contracts
        """
        if not self.filter_enabled:
            return daily_by_contract, []
        
        filtered_contracts = {}
        all_metrics = []
        
        for contract_name, df in daily_by_contract.items():
            # Skip non-HG contracts if commodity filter is set
            if self.commodity and not contract_name.startswith(self.commodity):
                filtered_contracts[contract_name] = df
                continue
            
            # Analyze quality
            metrics = self.analyze_contract_quality(df, contract_name)
            all_metrics.append(metrics)
            
            if metrics['status'] == 'INCLUDED':
                # Optionally trim early sparse data
                if self.trim_early_sparse:
                    df_trimmed = self.trim_early_sparse_data(df)
                    if len(df_trimmed) < len(df):
                        metrics['trimmed_points'] = len(df) - len(df_trimmed)
                        metrics['trimmed_from'] = df.index.min().strftime('%Y-%m-%d')
                        metrics['trimmed_to'] = df_trimmed.index.min().strftime('%Y-%m-%d')
                    filtered_contracts[contract_name] = df_trimmed
                else:
                    filtered_contracts[contract_name] = df
        
        excluded_info = [m for m in all_metrics if m['status'] == 'EXCLUDED']
        included_info = [m for m in all_metrics if m['status'] == 'INCLUDED']
        
        print(f"\nData Quality Filter Results:")
        print(f"  Total contracts: {len(daily_by_contract)}")
        print(f"  Excluded: {len(excluded_info)}")
        print(f"  Included: {len(included_info)}")
        
        return filtered_contracts, all_metrics
    
    def save_exclusion_report(self, metrics: List[dict], output_dir: Path):
        """Save detailed exclusion report and quality metrics."""
        output_dir = Path(output_dir)
        quality_dir = output_dir / 'data_quality'
        quality_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate excluded and included
        excluded = [m for m in metrics if m['status'] == 'EXCLUDED']
        included = [m for m in metrics if m['status'] == 'INCLUDED']
        
        # Save markdown report
        md_path = quality_dir / f'{self.commodity}_excluded_contracts.md'
        with open(md_path, 'w') as f:
            f.write(f'# {self.commodity} Data Quality Filtering Report\n\n')
            f.write(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # Configuration
            f.write('## Filtering Configuration\n\n')
            f.write(f'- **Cutoff Year**: {self.cutoff_year}\n')
            f.write(f'- **Min Data Points**: {self.min_data_points:,}\n')
            f.write(f'- **Max Gap Days**: {self.max_gap_days}\n')
            f.write(f'- **Min Coverage**: {self.min_coverage_percent}%\n')
            f.write(f'- **Trim Early Sparse**: {self.trim_early_sparse}\n\n')
            
            # Summary
            f.write('## Summary\n\n')
            f.write(f'- **Total Contracts**: {len(metrics)}\n')
            f.write(f'- **Excluded**: {len(excluded)}\n')
            f.write(f'- **Included**: {len(included)}\n')
            f.write(f'- **Exclusion Rate**: {len(excluded)/len(metrics)*100:.1f}%\n\n')
            
            # Excluded contracts
            f.write('## Excluded Contracts\n\n')
            if excluded:
                f.write('| Contract | Year | Data Points | Date Range | Coverage | Max Gap | Reasons |\n')
                f.write('|----------|------|-------------|------------|----------|---------|----------|\n')
                
                for m in sorted(excluded, key=lambda x: x['contract']):
                    reasons = '<br>'.join(m['reasons'])
                    f.write(f"| {m['contract']} | {m['year']} | {m['data_points']:,} | ")
                    f.write(f"{m['date_range']} | {m['coverage_percent']}% | ")
                    f.write(f"{m['max_gap_days']} days | {reasons} |\n")
            else:
                f.write('No contracts excluded.\n')
            
            # Included contracts with trimming
            f.write('\n## Included Contracts\n\n')
            trimmed_contracts = [m for m in included if 'trimmed_points' in m]
            
            if trimmed_contracts:
                f.write('### Contracts with Early Data Trimmed\n\n')
                f.write('| Contract | Original Points | Trimmed Points | Trimmed Period |\n')
                f.write('|----------|----------------|----------------|----------------|\n')
                
                for m in sorted(trimmed_contracts, key=lambda x: x['contract']):
                    orig_points = m['data_points'] + m.get('trimmed_points', 0)
                    f.write(f"| {m['contract']} | {orig_points:,} | {m['trimmed_points']:,} | ")
                    f.write(f"{m['trimmed_from']} to {m['trimmed_to']} |\n")
            
            f.write(f'\n### Total Included: {len(included)} contracts\n')
            
            # Gap analysis
            contracts_with_gaps = [m for m in excluded if m['gaps']]
            if contracts_with_gaps:
                f.write('\n## Detailed Gap Analysis\n\n')
                for m in sorted(contracts_with_gaps, key=lambda x: x['max_gap_days'], reverse=True)[:10]:
                    f.write(f"### {m['contract']}\n")
                    for gap in m['gaps']:
                        f.write(f"- {gap['start']} to {gap['end']}: **{gap['days']} days**\n")
                    f.write('\n')
        
        # Save CSV with all metrics
        csv_path = quality_dir / f'{self.commodity}_quality_metrics.csv'
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(csv_path, index=False)
        
        # Save JSON summary
        json_path = quality_dir / f'{self.commodity}_filtering_summary.json'
        summary = {
            'timestamp': datetime.now().isoformat(),
            'commodity': self.commodity,
            'configuration': {
                'cutoff_year': self.cutoff_year,
                'min_data_points': self.min_data_points,
                'max_gap_days': self.max_gap_days,
                'min_coverage_percent': self.min_coverage_percent,
                'trim_early_sparse': self.trim_early_sparse
            },
            'results': {
                'total_contracts': len(metrics),
                'excluded': len(excluded),
                'included': len(included),
                'exclusion_rate': round(len(excluded)/len(metrics)*100, 1)
            },
            'excluded_contracts': [m['contract'] for m in excluded],
            'included_contracts': [m['contract'] for m in included]
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nQuality reports saved:")
        print(f"  - {md_path}")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")
        
        return summary
