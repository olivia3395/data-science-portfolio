from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_json


def evaluate_policy_candidates(policy_df: pd.DataFrame, selected_df: pd.DataFrame, treatment_col: str, output_path: str | Path) -> dict:
    df = policy_df.copy()
    chosen = set(selected_df['action_id'].tolist()) if not selected_df.empty else set()
    df['policy_take_action'] = df['action_id'].isin(chosen).astype(int)

    p = df['propensity'].clip(1e-3, 1 - 1e-3)
    dr_value = np.where(
        df['policy_take_action'] == 1,
        df['mu1_hat'] + (df[treatment_col] == 1) * (df['target'] - df['mu1_hat']) / p,
        df['mu0_hat'] + (df[treatment_col] == 0) * (df['target'] - df['mu0_hat']) / (1 - p),
    )
    baseline_value = df['mu0_hat']
    summary = {
        'policy_value_dr': float(np.mean(dr_value)),
        'baseline_value_mu0': float(np.mean(baseline_value)),
        'estimated_uplift_vs_baseline': float(np.mean(dr_value) - np.mean(baseline_value)),
        'selection_rate': float(df['policy_take_action'].mean()),
    }
    save_json(summary, output_path)
    return summary
