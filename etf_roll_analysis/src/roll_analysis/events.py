import numpy as np
import pandas as pd


def detect_widening(spread: pd.Series, method: str = "zscore", window: int = 30,
                    z_threshold: float = 1.5, abs_min: float = None, 
                    cool_down: int = 0) -> pd.Series:
    """
    Detect spread widening events with multiple methods.
    
    Parameters:
    -----------
    spread : pd.Series
        Calendar spread series (next - front)
    method : str
        Detection method: "zscore", "abs", or "combined"
    window : int
        Rolling window for z-score calculation
    z_threshold : float
        Z-score threshold (e.g., 1.5 = 1.5 standard deviations)
    abs_min : float
        Minimum absolute change threshold
    cool_down : int
        Minimum days between events (0 = no cool-down)
    
    Returns:
    --------
    pd.Series
        Boolean series indicating widening events
    """
    # Calculate daily spread changes
    ds = spread.diff()
    
    if method == "zscore":
        # Z-score method: standardized changes
        # Use min_periods=max(5, window//2) to handle NaN values better
        min_periods = max(5, window // 2)
        mu = ds.rolling(window, min_periods=min_periods).mean()
        sd = ds.rolling(window, min_periods=min_periods).std()
        
        # Avoid division by zero
        sd = sd.replace(0, np.nan)
        z = (ds - mu) / sd
        
        # Detect when z-score exceeds threshold (positive changes only for widening)
        trigger = z > z_threshold
        
        # Optional: also require minimum absolute change
        if abs_min is not None:
            trigger &= (ds > abs_min)
            
    elif method == "abs":
        # Absolute threshold method
        if abs_min is None:
            raise ValueError("abs_min is required when method='abs'")
        trigger = ds > abs_min
        
    elif method == "combined":
        # Combined method: either z-score OR absolute threshold
        if abs_min is None:
            raise ValueError("abs_min is required when method='combined'")
        
        # Use min_periods=max(5, window//2) to handle NaN values better
        min_periods = max(5, window // 2)
        mu = ds.rolling(window, min_periods=min_periods).mean()
        sd = ds.rolling(window, min_periods=min_periods).std()
        sd = sd.replace(0, np.nan)
        z = (ds - mu) / sd
        
        trigger = (z > z_threshold) | (ds > abs_min)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply cool-down period if specified
    if cool_down > 0:
        trigger_filtered = trigger.copy()
        last_event_idx = -cool_down - 1
        
        for i in range(len(trigger)):
            if trigger.iloc[i]:
                if i - last_event_idx > cool_down:
                    last_event_idx = i
                else:
                    trigger_filtered.iloc[i] = False
        
        trigger = trigger_filtered
    
    # Fill NaN values with False
    trigger = trigger.fillna(False)
    
    return trigger.rename("widening_event")


def detect_narrowing(spread: pd.Series, **kwargs) -> pd.Series:
    """
    Detect spread narrowing events (opposite of widening).
    Uses same parameters as detect_widening but looks for negative changes.
    """
    # Invert the spread and detect widening
    return detect_widening(-spread, **kwargs).rename("narrowing_event")


def summarize_events(events: pd.Series, spread: pd.Series) -> pd.DataFrame:
    """
    Create summary table of widening/narrowing events.
    
    Returns DataFrame with:
    - date: event date
    - spread_before: spread value day before event
    - spread_after: spread value on event day
    - change: spread change
    - days_since_last: days since previous event
    """
    event_dates = events[events].index
    
    if len(event_dates) == 0:
        return pd.DataFrame(columns=['date', 'spread_before', 'spread_after', 'change', 'days_since_last'])
    
    summary = []
    prev_date = None
    
    for date in event_dates:
        idx = spread.index.get_loc(date)
        
        spread_before = spread.iloc[idx - 1] if idx > 0 else np.nan
        spread_after = spread.iloc[idx]
        change = spread_after - spread_before
        
        days_since = (date - prev_date).days if prev_date is not None else np.nan
        
        summary.append({
            'date': date,
            'spread_before': spread_before,
            'spread_after': spread_after,
            'change': change,
            'days_since_last': days_since
        })
        
        prev_date = date
    
    return pd.DataFrame(summary)
