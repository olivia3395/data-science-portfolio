# Data Science Portfolio

My applied data science projects: causal inference, forecasting, machine learning, NLP, computer vision and interactive apps. Each folder is a self-contained project with its own README. Every project was developed as a separate repository and moved here with its full commit history.

## Causal inference

| Project | Description |
|---|---|
| [Congestion-Pricing](causal-inference/Congestion-Pricing/) | Uses 12M+ NYC TLC trips to estimate the effect of the 2025 MTA congestion pricing, with TWFE DiD, Callaway–Sant'Anna, synthetic DiD, double ML and causal forests. |
| [CausalLens](causal-inference/CausalLens/) | A causal inference pipeline (DoWhy, EconML double ML, causal forest) with refutation tests and LLM-generated reports. |
| [Causal-Promotion-Optimization](causal-inference/Causal-Promotion-Optimization/) | Estimates promotion uplift on Rossmann data with doubly robust and CATE models, then optimizes promotions under a budget. |
| [AB-Testing](causal-inference/AB-Testing/) | A/B test analysis for ad targeting, with bootstrap confidence intervals and power analysis. |

## Forecasting

| Project | Description |
|---|---|
| [Retail-Demand-Forecasting](forecasting/Retail-Demand-Forecasting/) | M5 store-SKU demand forecasting: seasonal naive, LightGBM and a Temporal Fusion Transformer. |

## Statistics

| Project | Description |
|---|---|
| [Bayesian-Lung-Cancer-Prediction](statistics/Bayesian-Lung-Cancer-Prediction/) | Bayesian logistic regression with spike-and-slab variable selection, fit by MCMC in RStan. |

## Machine learning

| Project | Description |
|---|---|
| [Customer-Churn-Prediction](ml/Customer-Churn-Prediction/) | An MLOps pipeline: scikit-learn, MLflow, a FastAPI service, Evidently drift monitoring and Docker Compose. |
| [Traffic-Bot-Detection](ml/Traffic-Bot-Detection/) | Multi-layer bot detection for e-commerce traffic, combining statistical rules, Isolation Forest, gradient boosting and LLM behavioural fingerprints. |
| [Credit-Default-Risk](ml/Credit-Default-Risk/) | Credit default prediction with XGBoost and SMOTE. |
| [Customer-Segmentation](ml/Customer-Segmentation/) | RFM-based customer segmentation with K-means. |
| [VAE-NormalizingFlows](ml/VAE-NormalizingFlows/) | A VAE with normalizing flows and a multivariate g-and-k prior for MNIST. |
| [Village-Graph-Analysis](ml/Village-Graph-Analysis/) | Research code from my research assistantship at Tsinghua University. |
| [COVID-19-Regression](ml/COVID-19-Regression/) | Multiple linear regression on COVID-19 data. |

## NLP

| Project | Description |
|---|---|
| [Financial-Sentiment](nlp/Financial-Sentiment/) | Financial news sentiment classification with DistilBERT. |
| [Spam-Detection](nlp/Spam-Detection/) | Spam detection with TF-IDF features, K-means and Isolation Forest. |
| [RB-Lyrics](nlp/RB-Lyrics/) | Text analysis of R&B lyrics. |

## Computer vision

| Project | Description |
|---|---|
| [Dog-Breed-Classification](vision/Dog-Breed-Classification/) | Dog breed identification with VGG16 and ResNet50 transfer learning. |
| [Mask-Detection](vision/Mask-Detection/) | Face mask detection with ResNet50 and Grad-CAM. |

## Apps

| Project | Description |
|---|---|
| [DataPilot-Studio](apps/DataPilot-Studio/) | An AI data science agent for autonomous CSV analysis (Streamlit). |
| [Airbnb-Shiny-App](apps/Airbnb-Shiny-App/) | An R Shiny app for exploring Airbnb listings. |
| [Global-Risk-Analysis-USA](apps/Global-Risk-Analysis-USA/) | Interactive global risk analysis. |
| [School-Search](apps/School-Search/) | School search app. |
