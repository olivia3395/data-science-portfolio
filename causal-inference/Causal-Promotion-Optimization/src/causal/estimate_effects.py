from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.causal.diagnostics import compute_balance_table, overlap_summary
from src.evaluation.metrics import regression_metrics
from src.utils.io import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CausalArtifacts:
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    propensity_model: Any
    mu0_model: Any
    mu1_model: Any
    cate_model: Any | None
    use_econml: bool


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([
        ('num', numeric, numeric_cols),
        ('cat', categorical, categorical_cols),
    ])


def _split_holdout(df: pd.DataFrame, date_col: str, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff].copy()
    holdout = df[df[date_col] > cutoff].copy()
    return train, holdout


def _fit_dr_models(train_df: pd.DataFrame, holdout_df: pd.DataFrame, feature_cols: list[str], numeric_cols: list[str], categorical_cols: list[str], treatment_col: str, outcome_col: str, nuisance_params: dict, min_propensity: float, max_propensity: float, use_econml_if_available: bool = True):
    X_train = train_df[feature_cols]
    X_holdout = holdout_df[feature_cols]
    t_train = train_df[treatment_col].astype(int)
    y_train = train_df[outcome_col].astype(float)

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    propensity = Pipeline([
        ('prep', preprocessor),
        ('model', LGBMClassifier(objective='binary', random_state=42, **nuisance_params)),
    ])
    propensity.fit(X_train, t_train)

    mu0_model = Pipeline([
        ('prep', _build_preprocessor(numeric_cols, categorical_cols)),
        ('model', LGBMRegressor(objective='regression', random_state=42, **nuisance_params)),
    ])
    mu1_model = Pipeline([
        ('prep', _build_preprocessor(numeric_cols, categorical_cols)),
        ('model', LGBMRegressor(objective='regression', random_state=42, **nuisance_params)),
    ])
    mu0_model.fit(X_train[t_train == 0], y_train[t_train == 0])
    mu1_model.fit(X_train[t_train == 1], y_train[t_train == 1])

    p_train = propensity.predict_proba(X_train)[:, 1].clip(min_propensity, max_propensity)
    p_holdout = propensity.predict_proba(X_holdout)[:, 1].clip(min_propensity, max_propensity)
    mu0_holdout = mu0_model.predict(X_holdout)
    mu1_holdout = mu1_model.predict(X_holdout)

    cate_holdout = mu1_holdout - mu0_holdout
    cate_model = None
    use_econml = False

    if use_econml_if_available:
        try:
            from econml.dr import DRLearner
            logger.info('Using econml.DRLearner for CATE estimation.')
            cate_model = DRLearner(
                model_propensity=LGBMClassifier(objective='binary', random_state=42, **nuisance_params),
                model_regression=LGBMRegressor(objective='regression', random_state=42, **nuisance_params),
                model_final=LGBMRegressor(objective='regression', random_state=42, **nuisance_params),
            )
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_holdout_transformed = preprocessor.transform(X_holdout)
            cate_model.fit(y_train.to_numpy(), t_train.to_numpy(), X=X_train_transformed)
            cate_holdout = cate_model.effect(X_holdout_transformed)
            use_econml = True
        except Exception as exc:
            logger.warning('econml DRLearner unavailable; falling back to T-learner style CATE: %s', exc)

    artifacts = CausalArtifacts(
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        propensity_model=propensity,
        mu0_model=mu0_model,
        mu1_model=mu1_model,
        cate_model=cate_model,
        use_econml=use_econml,
    )
    return artifacts, p_train, p_holdout, mu0_holdout, mu1_holdout, cate_holdout


def estimate_causal_effects(df: pd.DataFrame, cfg: dict, artifact_dir: str | Path) -> tuple[pd.DataFrame, dict, CausalArtifacts, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    treatment_col = cfg['features']['treatment_col']
    outcome_col = 'target'
    holdout_days = int(cfg['data']['holdout_days'])

    excluded = {'Date', 'target', 'Sales', 'Customers'}
    feature_cols = [c for c in df.columns if c not in excluded]
    categorical_cols = [c for c in feature_cols if str(df[c].dtype) in ('object', 'category')]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    train_df, holdout_df = _split_holdout(df, 'Date', holdout_days)
    artifacts, p_train, p_holdout, mu0_holdout, mu1_holdout, cate_holdout = _fit_dr_models(
        train_df,
        holdout_df,
        feature_cols,
        numeric_cols,
        categorical_cols,
        treatment_col,
        outcome_col,
        cfg['causal']['nuisance_model'],
        cfg['causal']['min_propensity'],
        cfg['causal']['max_propensity'],
        cfg['causal']['use_econml_if_available'],
    )

    holdout = holdout_df.copy()
    holdout['propensity'] = p_holdout
    holdout['mu0_hat'] = mu0_holdout
    holdout['mu1_hat'] = mu1_holdout
    holdout['cate_hat'] = cate_holdout
    holdout['aipw_score'] = (
        holdout['mu1_hat'] - holdout['mu0_hat']
        + holdout[treatment_col] * (holdout[outcome_col] - holdout['mu1_hat']) / holdout['propensity']
        - (1 - holdout[treatment_col]) * (holdout[outcome_col] - holdout['mu0_hat']) / (1 - holdout['propensity'])
    )

    ate = float(holdout['aipw_score'].mean())
    att = float(holdout.loc[holdout[treatment_col] == 1, 'aipw_score'].mean())
    auc = float(roc_auc_score(train_df[treatment_col], p_train)) if train_df[treatment_col].nunique() > 1 else float('nan')
    outcome_pred = np.where(holdout[treatment_col] == 1, holdout['mu1_hat'], holdout['mu0_hat'])
    metrics = {
        'ate_aipw': ate,
        'att_aipw': att,
        'propensity_auc_train': auc,
        'factual_outcome_fit_holdout': regression_metrics(holdout[outcome_col], outcome_pred),
        'cate_mean_holdout': float(np.mean(cate_holdout)),
        'cate_p90_holdout': float(np.quantile(cate_holdout, 0.9)),
    }

    balance = compute_balance_table(holdout, treatment_col=treatment_col, feature_cols=numeric_cols, propensity_col='propensity')
    overlap = overlap_summary(holdout, treatment_col=treatment_col, propensity_col='propensity')
    metrics['overlap'] = overlap

    out_dir = Path(artifact_dir)
    save_json(metrics, out_dir / 'metrics' / 'causal_metrics.json')
    balance.to_csv(out_dir / 'metrics' / 'balance_table.csv', index=False)
    holdout.to_csv(out_dir / 'predictions' / 'holdout_cate_predictions.csv', index=False)

    feature_importance_rows = []
    try:
        model = artifacts.propensity_model.named_steps['model']
        prep = artifacts.propensity_model.named_steps['prep']
        names = prep.get_feature_names_out()
        importances = model.feature_importances_
        feature_importance_rows = pd.DataFrame({'feature': names, 'importance': importances}).sort_values('importance', ascending=False)
        feature_importance_rows.to_csv(out_dir / 'metrics' / 'propensity_feature_importance.csv', index=False)
    except Exception as exc:
        logger.warning('Could not export propensity feature importance: %s', exc)
        feature_importance_rows = pd.DataFrame(columns=['feature', 'importance'])

    return holdout, metrics, artifacts, balance, feature_importance_rows
