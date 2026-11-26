from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_features(panel_path: str | Path, output_path: str | Path, lag_days: list[int], rolling_windows: list[int], clip_negative_sales: bool = True, log1p_target: bool = False, drop_closed_days: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(panel_path).sort_values(['Store', 'Date']).copy()
    df['Date'] = pd.to_datetime(df['Date'])

    if clip_negative_sales:
        df['Sales'] = df['Sales'].clip(lower=0)
    if drop_closed_days:
        df = df[df['Open'] == 1].copy()

    grouped = df.groupby('Store', sort=False)
    for lag in lag_days:
        df[f'sales_lag_{lag}'] = grouped['Sales'].shift(lag)
        df[f'customers_lag_{lag}'] = grouped['Customers'].shift(lag)
        df[f'promo_lag_{lag}'] = grouped['Promo'].shift(lag)

    for window in rolling_windows:
        df[f'sales_roll_mean_{window}'] = grouped['Sales'].shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        df[f'sales_roll_std_{window}'] = grouped['Sales'].shift(1).rolling(window).std().reset_index(level=0, drop=True)
        df[f'customers_roll_mean_{window}'] = grouped['Customers'].shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        df[f'promo_roll_mean_{window}'] = grouped['Promo'].shift(1).rolling(window).mean().reset_index(level=0, drop=True)

    def _days_since_last(values: pd.Series) -> pd.Series:
        days = []
        counter = 999
        prev = 0
        for val in values.shift(1).fillna(0).astype(int):
            if prev == 1:
                counter = 1
            else:
                counter = counter + 1 if counter < 999 else 999
            days.append(counter)
            prev = int(val)
        return pd.Series(days, index=values.index)

    df['days_since_last_promo'] = grouped['Promo'].transform(_days_since_last)

    # Seasonal encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

    if log1p_target:
        df['target'] = np.log1p(df['Sales'])
    else:
        df['target'] = df['Sales']

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[f'sales_lag_{lag_days[-1]}']).reset_index(drop=True)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    df.to_parquet(output_path, index=False)
    logger.info('Saved featured dataset to %s with %s rows', output_path, len(df))
    return df
