<div align="center">

# 🧠 Causal Promotion Optimization for Retail Demand

### Estimate Incremental Promotion Lift and Allocate Budget Like a Pricing & Promotions Science Team

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Nuisance_Models-9ACD32?style=flat-square)
![EconML](https://img.shields.io/badge/EconML-DRLearner-185FA5?style=flat-square)
![DoWhy](https://img.shields.io/badge/DoWhy-Refutation-1D9E75?style=flat-square)
![OR--Tools](https://img.shields.io/badge/OR--Tools-Budget_Optimization-F97316?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-Rossmann-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

**A production-style causal ML project for retail promotions.**  
This repo downloads a real retail dataset, builds a store-day panel, estimates promotion uplift with doubly robust causal models, runs diagnostics and refutation tests, and converts uplift estimates into budget-constrained promotion recommendations.

</div>



## ✨ What this project does

This repository is built to feel closer to a real **pricing & promotions science** workflow than a classroom causal inference notebook.

It covers the full loop:

1. **download retail data**
2. **build a panel dataset**
3. **estimate causal promotion effects**
4. **check overlap and balance**
5. **run causal refutation tests**
6. **rank candidate promotions by expected uplift**
7. **solve a budget allocation problem**
8. **serve recommendations through an API and dashboard**



## 🏪 Business framing

Retail teams do not usually ask:

> "Do promotions help sales?"

They ask:

> "Given a limited budget, which stores and time periods should receive promotions, and what incremental revenue or margin should we expect?"

This project is designed around that question.

### Inputs

- historical sales
- customer traffic
- promo exposure
- calendar effects
- school and state holidays
- store metadata
- competition distance and timing
- lagged demand and rolling demand signals

### Outputs

- **ATE**: average treatment effect of promotion
- **ATT**: effect on promoted observations
- **CATE**: heterogeneous store-day uplift estimates
- **uplift-ranked promotion candidates**
- **budget-constrained promotion recommendations**
- **offline policy value estimates**



## 🗂️ Dataset

This project defaults to the **Rossmann Store Sales** retail dataset from Kaggle. Rossmann is a well-known forecasting benchmark containing daily sales observations for 1,115 German drug stores, plus promotion, competition, and store metadata.



Rossmann is especially useful for a pricing & promotions project because it includes:

- a direct promotion indicator (`Promo`)
- rich time variation
- store heterogeneity
- holiday structure
- store-level context variables

### Core files used by this repo

#### `train.csv`
Store-day observations including:
- `Store`
- `DayOfWeek`
- `Date`
- `Sales`
- `Customers`
- `Open`
- `Promo`
- `StateHoliday`
- `SchoolHoliday`

#### `store.csv`
Store-level metadata including:
- `StoreType`
- `Assortment`
- `CompetitionDistance`
- `CompetitionOpenSinceMonth`
- `CompetitionOpenSinceYear`
- `Promo2`
- `Promo2SinceWeek`
- `Promo2SinceYear`
- `PromoInterval`

### Download behavior

The project tries these paths automatically:

1. Kaggle competition download for `rossmann-store-sales`
2. Kaggle API competition download
3. Kaggle dataset fallback: `shahpranshu27/rossman-store-sales`
4. Kaggle API dataset fallback





## 🏗️ Project structure

```text
causal_promotion_optimization/
├── configs/
│   └── default.yaml
├── scripts/
│   └── bootstrap.sh
├── src/
│   ├── data/
│   │   ├── download_dataset.py
│   │   └── build_panel.py
│   ├── features/
│   │   └── build_features.py
│   ├── causal/
│   │   ├── estimate_effects.py
│   │   ├── diagnostics.py
│   │   └── dowhy_refute.py
│   ├── optimization/
│   │   └── solve_budget_plan.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── policy_evaluation.py
│   ├── serving/
│   │   └── app.py
│   ├── dashboard/
│   │   └── app.py
│   ├── utils/
│   │   ├── io.py
│   │   └── logging_utils.py
│   └── run_pipeline.py
├── artifacts/
├── requirements.txt
└── README.md
```



## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```



## 🔐 Kaggle authentication

### Option A: `kaggle.json`

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Option B: environment variables

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

For competition downloads, you must also have accepted the competition rules on Kaggle.


## 🚀 One-command bootstrap

```bash
bash scripts/bootstrap.sh
```

This will:

1. download Rossmann data
2. build a store-day panel
3. engineer causal and forecasting covariates
4. estimate promotion lift
5. export diagnostics and refutation results
6. build a promotion candidate table
7. solve a budget allocation plan
8. save artifacts for API and dashboard use


## 🧠 Causal modeling approach

This repo treats promotion assignment as **observational**, not randomized.

### Treatment

- `Promo = 1` if a store-day is promoted
- `Promo = 0` otherwise

### Outcome

- default: `Sales`
- optional: log-transformed sales through config

### Identification strategy

We assume promotion exposure is conditionally ignorable given observed covariates such as:

- lagged sales and customers
- rolling demand statistics
- holiday and school holiday indicators
- store type and assortment
- competition distance and age
- time-of-week and month effects

### Estimation

The pipeline fits:

- a **propensity model**
- two **outcome models** for treatment and control
- an **AIPW / doubly robust effect estimate**
- an optional **EconML DRLearner** CATE model when available

### Diagnostics

- propensity overlap summary
- covariate balance table
- optional DoWhy refutation tests



## 💸 Budget optimization

After estimating uplift, the project converts each candidate action into a business table with:

- expected incremental revenue
- proxy promotion cost
- proxy incremental margin
- ROI proxy

It then solves a budget-constrained selection problem using **OR-Tools**, with a greedy fallback if the solver is unavailable.

> Note: Rossmann does not include observed discount percentages or true promotion costs, so this repo uses configurable cost and margin proxies. Those assumptions are explicit and easy to replace with real business inputs.




## 🔌 API

Start the API with:

```bash
uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health`
- `POST /score` — returns propensity, mu0, mu1, and CATE for candidate rows
- `POST /optimize` — scores candidates and returns a budget-constrained action plan



## 📊 Dashboard

Start the dashboard with:

```bash
streamlit run src/dashboard/app.py
```

The dashboard shows:

- ATE / ATT / mean CATE
- CATE distribution by treatment status
- top recommended actions
- cost vs margin view of selected promotions



<div align="center">

</div>
