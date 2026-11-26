from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def standardized_mean_difference(x_treated: np.ndarray, x_control: np.ndarray, weights_treated: np.ndarray | None = None, weights_control: np.ndarray | None = None) -> float:
    if weights_treated is None:
        mu_t = np.mean(x_treated)
        var_t = np.var(x_treated)
    else:
        mu_t = np.average(x_treated, weights=weights_treated)
        var_t = np.average((x_treated - mu_t) ** 2, weights=weights_treated)

    if weights_control is None:
        mu_c = np.mean(x_control)
        var_c = np.var(x_control)
    else:
        mu_c = np.average(x_control, weights=weights_control)
        var_c = np.average((x_control - mu_c) ** 2, weights=weights_control)

    pooled = np.sqrt((var_t + var_c) / 2 + 1e-9)
    return float((mu_t - mu_c) / pooled)


def compute_balance_table(df: pd.DataFrame, treatment_col: str, feature_cols: Sequence[str], propensity_col: str = 'propensity') -> pd.DataFrame:
    treated = df[treatment_col] == 1
    control = ~treated
    ipw_t = 1.0 / df.loc[treated, propensity_col].clip(1e-3, 1 - 1e-3)
    ipw_c = 1.0 / (1.0 - df.loc[control, propensity_col].clip(1e-3, 1 - 1e-3))

    rows = []
    for col in feature_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        smd_raw = standardized_mean_difference(df.loc[treated, col].to_numpy(), df.loc[control, col].to_numpy())
        smd_weighted = standardized_mean_difference(
            df.loc[treated, col].to_numpy(),
            df.loc[control, col].to_numpy(),
            ipw_t.to_numpy(),
            ipw_c.to_numpy(),
        )
        rows.append({'feature': col, 'smd_raw': smd_raw, 'smd_ipw': smd_weighted, 'abs_smd_raw': abs(smd_raw), 'abs_smd_ipw': abs(smd_weighted)})
    return pd.DataFrame(rows).sort_values('abs_smd_raw', ascending=False)


def overlap_summary(df: pd.DataFrame, treatment_col: str, propensity_col: str = 'propensity') -> dict[str, float]:
    p = df[propensity_col].clip(1e-6, 1 - 1e-6)
    treated = df[treatment_col] == 1
    control = ~treated
    return {
        'treated_propensity_mean': float(p[treated].mean()),
        'control_propensity_mean': float(p[control].mean()),
        'propensity_p01': float(p.quantile(0.01)),
        'propensity_p99': float(p.quantile(0.99)),
        'share_low_overlap': float(((p < 0.05) | (p > 0.95)).mean()),
    }
