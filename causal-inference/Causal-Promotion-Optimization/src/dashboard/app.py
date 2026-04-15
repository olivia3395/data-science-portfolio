from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts'

st.set_page_config(page_title='Causal Promotion Optimization', layout='wide')
st.title('🧠 Causal Promotion Optimization Dashboard')

metrics_path = ART / 'metrics' / 'causal_metrics.json'
plan_path = ART / 'predictions' / 'recommended_promotion_plan.csv'
holdout_path = ART / 'predictions' / 'holdout_cate_predictions.csv'

if not metrics_path.exists():
    st.warning('Run the pipeline first: python -m src.run_pipeline --config configs/default.yaml')
    st.stop()

with open(metrics_path, 'r', encoding='utf-8') as f:
    metrics = json.load(f)
holdout = pd.read_csv(holdout_path)
plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()

c1, c2, c3 = st.columns(3)
c1.metric('ATE (AIPW)', f"{metrics['ate_aipw']:.2f}")
c2.metric('ATT (AIPW)', f"{metrics['att_aipw']:.2f}")
c3.metric('Mean CATE', f"{metrics['cate_mean_holdout']:.2f}")

st.subheader('CATE distribution')
fig = px.histogram(holdout, x='cate_hat', nbins=50, color='Promo', marginal='box')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Top recommended promotion actions')
if not plan.empty:
    st.dataframe(plan[['Store', 'Date', 'cate_hat', 'incremental_margin_proxy', 'promo_cost_proxy', 'roi_proxy']].head(100), use_container_width=True)
    fig2 = px.scatter(plan.head(300), x='promo_cost_proxy', y='incremental_margin_proxy', color='roi_proxy', hover_data=['Store', 'Date'])
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info('No plan found yet.')
