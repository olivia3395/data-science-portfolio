from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _parse_store_competition_months(df: pd.DataFrame) -> pd.DataFrame:
    months = (df['CompetitionOpenSinceYear'].fillna(0).astype(int) - 1900) * 12 + df['CompetitionOpenSinceMonth'].fillna(1).astype(int)
    sale_months = (df['Date'].dt.year - 1900) * 12 + df['Date'].dt.month
    elapsed = sale_months - months
    df['competition_open_months'] = elapsed.where(elapsed >= 0, 0)
    return df


def build_rossmann_panel(raw_dir: str | Path, processed_dir: str | Path, train_file: str = 'train.csv', store_file: str = 'store.csv', sample_n_rows: int | None = None, min_date: str | None = None, max_date: str | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    processed_dir = ensure_dir(processed_dir)

    train = pd.read_csv(raw_dir / train_file, low_memory=False)
    store = pd.read_csv(raw_dir / store_file, low_memory=False)

    train['Date'] = pd.to_datetime(train['Date'])
    if min_date is not None:
        train = train[train['Date'] >= pd.Timestamp(min_date)]
    if max_date is not None:
        train = train[train['Date'] <= pd.Timestamp(max_date)]
    if sample_n_rows is not None:
        train = train.sort_values('Date').tail(int(sample_n_rows)).copy()

    df = train.merge(store, on='Store', how='left')
    df['Promo'] = df['Promo'].fillna(0).astype(int)
    df['Promo2'] = df['Promo2'].fillna(0).astype(int)
    df['SchoolHoliday'] = df['SchoolHoliday'].fillna(0).astype(int)
    df['Open'] = df['Open'].fillna(1).astype(int)
    df['StateHoliday'] = df['StateHoliday'].fillna('0').astype(str)
    df['StoreType'] = df['StoreType'].fillna('unknown').astype(str)
    df['Assortment'] = df['Assortment'].fillna('unknown').astype(str)
    df['PromoInterval'] = df['PromoInterval'].fillna('').astype(str)
    df['Sales'] = df['Sales'].fillna(0.0).astype(float)
    df['Customers'] = df['Customers'].fillna(0.0).astype(float)
    df['promo2_active'] = 0

    promo_month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_name = df['Date'].dt.month.map(promo_month_map)
    df.loc[df['Promo2'] == 1, 'promo2_active'] = [
        int(m in str(interval).split(',')) if interval else 0
        for m, interval in zip(month_name, df['PromoInterval'])
    ]

    df = _parse_store_competition_months(df)
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['DayOfWeek'].astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([6, 7]).astype(int)

    # Proxy eligibility: we only optimize over days in which a promo could plausibly be run.
    df['promo_candidate'] = (df['Open'] == 1).astype(int)

    ordered_cols = [
        'Store', 'Date', 'Sales', 'Customers', 'Promo', 'Promo2', 'promo2_active', 'Open', 'SchoolHoliday',
        'StateHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'competition_open_months',
        'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'year', 'month', 'weekofyear', 'dayofweek',
        'is_month_start', 'is_month_end', 'is_weekend', 'promo_candidate'
    ]
    panel = df[ordered_cols].sort_values(['Store', 'Date']).reset_index(drop=True)
    panel.to_parquet(processed_dir / 'rossmann_panel.parquet', index=False)
    panel.to_csv(processed_dir / 'rossmann_panel.csv', index=False)
    logger.info('Saved panel with %s rows to %s', len(panel), processed_dir)
    return panel
