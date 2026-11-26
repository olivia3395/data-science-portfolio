from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.optimization.solve_budget_plan import build_policy_table, solve_budget_allocation
from src.utils.io import load_joblib, load_yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = load_yaml(ROOT / 'configs' / 'default.yaml')
ARTIFACT_PATH = ROOT / 'artifacts' / 'models' / 'causal_artifacts.joblib'

app = FastAPI(title='Causal Promotion Optimization API', version='1.0.0')


def _load_artifacts():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError('Missing artifacts/models/causal_artifacts.joblib. Run the training pipeline first.')
    return load_joblib(ARTIFACT_PATH)


class ScoreRequest(BaseModel):
    records: list[dict[str, Any]]


class OptimizeRequest(BaseModel):
    records: list[dict[str, Any]]
    budget: float | None = None


def _score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = _load_artifacts()
    feature_cols = artifacts['feature_cols']
    X = df[feature_cols]
    p = artifacts['propensity_model'].predict_proba(X)[:, 1]
    mu0 = artifacts['mu0_model'].predict(X)
    mu1 = artifacts['mu1_model'].predict(X)
    scored = df.copy()
    scored['propensity'] = p
    scored['mu0_hat'] = mu0
    scored['mu1_hat'] = mu1
    scored['cate_hat'] = mu1 - mu0
    return scored


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/score')
def score(request: ScoreRequest) -> dict[str, Any]:
    df = pd.DataFrame(request.records)
    scored = _score_dataframe(df)
    return {'predictions': scored.to_dict(orient='records')}


@app.post('/optimize')
def optimize(request: OptimizeRequest) -> dict[str, Any]:
    df = pd.DataFrame(request.records)
    scored = _score_dataframe(df)
    cfg = load_yaml(ROOT / 'configs' / 'default.yaml')
    if request.budget is not None:
        cfg['optimization']['budget'] = request.budget
    policy_df = build_policy_table(scored, cfg)
    selected, summary = solve_budget_allocation(policy_df, cfg, ROOT / 'artifacts')
    return {'summary': summary, 'selected_actions': selected.to_dict(orient='records')}
