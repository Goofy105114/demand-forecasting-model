# Production Demand Forecasting

A machine learning app that forecasts monthly production demand per warehouse × product category, built to support manufacturing planning.

**Live app →** *(add your Streamlit Cloud URL here after deploying)*

---

## What it does

- Forecasts demand up to 6 months ahead using recursive multi-step prediction
- Compares 5 ML models + 3 baselines, selected by rolling-origin cross-validation
- Includes a production planning view with configurable safety stock
- Deployed as a Streamlit web app

## Results

| Model | Hold-out WAPE % |
|---|---|
| 3-month Moving Average (best baseline) | 12.4% |
| **Gradient Boosting (champion)** | **11.8%** |
| Hist Gradient Boosting (best single window) | 10.3% |

Champion chosen by lowest mean WAPE across 3 rolling-origin CV folds.

## Method

```
Raw CSV (1M rows)
  → clean (fix bracketed negatives, drop bad dates, filter 2012–2016)
  → aggregate to monthly demand per warehouse × category (33 series)
  → lag / rolling / calendar features (no look-ahead)
  → time-based train/test split (last 6 months = test)
  → compare baselines + 5 sklearn models
  → rolling-origin cross-validation (3 folds)
  → hyperparameter tuning (grid search on HGB)
  → refit champion on full data → save artifacts
  → serve via Streamlit
```

## Run locally

```bash
pip install -r requirements.txt

# If you have the raw data (place at data/Historical Product Demand.csv):
python train.py          # regenerates artifacts/ (~1–2 min)

# Launch the app (artifacts/ already included in repo):
streamlit run app.py
```

## Project structure

```
app.py                  ← Streamlit application (entry point)
pipeline.py             ← shared pipeline: cleaning, features, models, metrics
train.py                ← training script; writes artifacts/
artifacts/              ← pre-trained models + result tables (committed to repo)
figures/                ← charts used in the report
requirements.txt
.streamlit/config.toml  ← Streamlit Cloud configuration
```

## Dataset

Kaggle — [Forecasts for Product Demand](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)  
1,048,575 order records · 4 warehouses · 33 product categories · Jan 2012–Dec 2016
