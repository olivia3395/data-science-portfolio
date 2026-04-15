from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_policy_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    discount_rate = float(cfg['optimization']['assumed_discount_rate'])
    margin_rate = float(cfg['optimization']['assumed_margin_rate'])
    out['baseline_revenue'] = out['mu0_hat'].clip(lower=0)
    out['incremental_revenue'] = out['cate_hat']
    out['promo_cost_proxy'] = (out['baseline_revenue'].clip(lower=0) * discount_rate).clip(lower=1.0)
    out['incremental_margin_proxy'] = out['incremental_revenue'] * margin_rate - out['promo_cost_proxy']
    out['roi_proxy'] = out['incremental_margin_proxy'] / out['promo_cost_proxy'].replace(0, np.nan)
    out['eligible_action'] = (out['promo_candidate'] == 1).astype(int)
    out['action_id'] = np.arange(len(out))
    return out


def solve_budget_allocation(policy_df: pd.DataFrame, cfg: dict, artifact_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    budget = float(cfg['optimization']['budget'])
    max_actions = int(cfg['optimization']['max_actions'])
    min_roi = float(cfg['optimization']['min_roi'])
    candidates = policy_df[(policy_df['eligible_action'] == 1) & (policy_df['roi_proxy'].fillna(-999) >= min_roi)].copy()

    if candidates.empty:
        summary = {'selected_actions': 0, 'total_cost': 0.0, 'total_incremental_margin_proxy': 0.0, 'total_incremental_revenue': 0.0}
        return candidates, summary

    try:
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if solver is None:
            raise RuntimeError('SCIP solver unavailable')
        x = {idx: solver.IntVar(0, 1, f'x_{idx}') for idx in candidates.index}
        solver.Add(sum(x[i] * float(candidates.loc[i, 'promo_cost_proxy']) for i in candidates.index) <= budget)
        solver.Add(sum(x[i] for i in candidates.index) <= max_actions)
        solver.Maximize(sum(x[i] * float(candidates.loc[i, 'incremental_margin_proxy']) for i in candidates.index))
        status = solver.Solve()
        selected_idx = [i for i in candidates.index if x[i].solution_value() > 0.5]
        logger.info('OR-Tools status: %s; selected %s actions', status, len(selected_idx))
    except Exception as exc:
        logger.warning('OR-Tools solver unavailable, using greedy fallback: %s', exc)
        ranked = candidates.sort_values('roi_proxy', ascending=False)
        running_cost = 0.0
        selected_idx = []
        for idx, row in ranked.iterrows():
            cost = float(row['promo_cost_proxy'])
            if running_cost + cost <= budget and len(selected_idx) < max_actions:
                selected_idx.append(idx)
                running_cost += cost

    selected = candidates.loc[selected_idx].sort_values('incremental_margin_proxy', ascending=False).copy()
    summary = {
        'selected_actions': int(len(selected)),
        'total_cost': float(selected['promo_cost_proxy'].sum()),
        'total_incremental_margin_proxy': float(selected['incremental_margin_proxy'].sum()),
        'total_incremental_revenue': float(selected['incremental_revenue'].sum()),
        'average_roi_proxy': float(selected['roi_proxy'].replace([np.inf, -np.inf], np.nan).fillna(0).mean()) if len(selected) else 0.0,
    }

    art_dir = Path(artifact_dir)
    selected.to_csv(art_dir / 'predictions' / 'recommended_promotion_plan.csv', index=False)
    save_json(summary, art_dir / 'metrics' / 'optimization_summary.json')
    return selected, summary
