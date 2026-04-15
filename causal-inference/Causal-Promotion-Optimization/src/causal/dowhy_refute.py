from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_dowhy_refuters(df: pd.DataFrame, treatment_col: str, outcome_col: str, feature_cols: list[str], output_path: str | Path, placebo_simulations: int = 30, random_common_cause_simulations: int = 30) -> dict:
    try:
        from dowhy import CausalModel
    except Exception as exc:
        logger.warning('DoWhy unavailable, skipping refuters: %s', exc)
        return {'status': 'skipped', 'reason': str(exc)}

    model_df = df[[treatment_col, outcome_col] + feature_cols].copy().dropna()
    graph = 'digraph { ' + ' '.join([f'{c} -> {treatment_col}; {c} -> {outcome_col};' for c in feature_cols]) + f' {treatment_col} -> {outcome_col}; }}'
    model = CausalModel(data=model_df, treatment=treatment_col, outcome=outcome_col, graph=graph)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified, method_name='backdoor.linear_regression')
    placebo = model.refute_estimate(identified, estimate, method_name='placebo_treatment_refuter', num_simulations=placebo_simulations)
    random_cc = model.refute_estimate(identified, estimate, method_name='random_common_cause', num_simulations=random_common_cause_simulations)
    result = {
        'status': 'ok',
        'estimate_value': float(estimate.value),
        'placebo': str(placebo),
        'random_common_cause': str(random_cc),
    }
    save_json(result, output_path)
    return result
