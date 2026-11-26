from __future__ import annotations

import argparse
from pathlib import Path

from src.causal.dowhy_refute import run_dowhy_refuters
from src.causal.estimate_effects import estimate_causal_effects
from src.data.build_panel import build_rossmann_panel
from src.data.download_dataset import ensure_rossmann_data
from src.evaluation.policy_evaluation import evaluate_policy_candidates
from src.features.build_features import build_features
from src.optimization.solve_budget_plan import build_policy_table, solve_budget_allocation
from src.utils.io import dump_joblib, ensure_dir, load_yaml, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    artifact_dir = Path(cfg['project']['artifact_dir'])
    for sub in ['data', 'metrics', 'predictions', 'models', 'processed', 'raw']:
        ensure_dir(artifact_dir / sub)

    ensure_rossmann_data(
        raw_dir=cfg['data']['raw_dir'],
        competition=cfg['data']['competition'],
        dataset_fallback=cfg['data']['dataset_fallback'],
    )

    panel = build_rossmann_panel(
        raw_dir=cfg['data']['raw_dir'],
        processed_dir=cfg['data']['processed_dir'],
        train_file=cfg['data']['train_file'],
        store_file=cfg['data']['store_file'],
        sample_n_rows=cfg['data']['sample_n_rows'],
        min_date=cfg['data']['min_date'],
        max_date=cfg['data']['max_date'],
    )

    featured = build_features(
        panel_path=Path(cfg['data']['processed_dir']) / 'rossmann_panel.parquet',
        output_path=cfg['data']['featured_path'],
        lag_days=cfg['features']['lag_days'],
        rolling_windows=cfg['features']['rolling_windows'],
        clip_negative_sales=cfg['features']['clip_negative_sales'],
        log1p_target=cfg['features']['log1p_target'],
        drop_closed_days=cfg['features']['drop_closed_days'],
    )

    holdout, metrics, artifacts, balance, fi = estimate_causal_effects(featured, cfg, artifact_dir)
    logger.info('Causal metrics: %s', metrics)

    policy_df = build_policy_table(holdout, cfg)
    selected, opt_summary = solve_budget_allocation(policy_df, cfg, artifact_dir)
    logger.info('Optimization summary: %s', opt_summary)

    policy_eval = evaluate_policy_candidates(policy_df, selected, treatment_col=cfg['features']['treatment_col'], output_path=artifact_dir / 'metrics' / 'policy_evaluation.json')
    logger.info('Policy evaluation: %s', policy_eval)

    if cfg['causal']['dowhy']['enabled']:
        sample = holdout.sample(n=min(int(cfg['causal']['dowhy']['sample_n']), len(holdout)), random_state=cfg['project']['random_seed']).copy()
        numeric_features = [c for c in artifacts.feature_cols if c in holdout.columns and str(holdout[c].dtype) != 'object'][:20]
        run_dowhy_refuters(
            sample,
            treatment_col=cfg['features']['treatment_col'],
            outcome_col='target',
            feature_cols=numeric_features,
            output_path=artifact_dir / 'metrics' / 'dowhy_refutation.json',
            placebo_simulations=int(cfg['causal']['dowhy']['placebo_simulations']),
            random_common_cause_simulations=int(cfg['causal']['dowhy']['random_common_cause_simulations']),
        )

    serializable_artifacts = {
        'feature_cols': artifacts.feature_cols,
        'numeric_cols': artifacts.numeric_cols,
        'categorical_cols': artifacts.categorical_cols,
        'propensity_model': artifacts.propensity_model,
        'mu0_model': artifacts.mu0_model,
        'mu1_model': artifacts.mu1_model,
        'use_econml': artifacts.use_econml,
    }
    dump_joblib(serializable_artifacts, artifact_dir / 'models' / 'causal_artifacts.joblib')
    save_json({'status': 'ok'}, artifact_dir / 'metrics' / 'run_status.json')


if __name__ == '__main__':
    main()
