"""
Production Demand Forecasting - core pipeline
Used by the notebook, the training script and the Streamlit app.

Granularity: monthly demand per (Warehouse, Product_Category) series.
Target: next-month demand (we forecast month t using only information up to t-1).
"""
import numpy as np
import pandas as pd

RAW_FILE = "data/Historical Product Demand.csv"
START, END = "2012-01-01", "2016-12-31"   # 2011 and 2017 are almost empty
LAGS = [1, 2, 3, 6, 12]
MIN_ACTIVE_MONTHS = 36   # a series needs enough history to be forecastable
MIN_TOTAL_DEMAND = 1_000_000


# ----------------------------------------------------------------- loading
def load_raw(path=RAW_FILE):
    return pd.read_csv(path, dtype=str)


def clean(raw):
    """Return (clean_df, report_dict) with every cleaning decision logged."""
    rep = {"raw_rows": len(raw)}
    df = raw.copy()
    df["Order_Demand"] = df["Order_Demand"].str.strip()
    neg = df["Order_Demand"].str.contains(r"\(", na=False)
    rep["bracketed_negative_values"] = int(neg.sum())
    df["Order_Demand"] = pd.to_numeric(
        df["Order_Demand"].str.replace(r"[()]", "", regex=True), errors="coerce")
    df.loc[neg, "Order_Demand"] *= -1          # (100) is accounting notation for -100
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    rep["missing_dates_dropped"] = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"])
    rep["exact_duplicate_rows"] = int(df.duplicated().sum())
    # exact duplicates kept: the same product can legitimately be ordered twice
    # on one day (checked: duplicates are overwhelmingly round order sizes).
    rep["negative_rows_kept_as_returns"] = int((df["Order_Demand"] < 0).sum())
    df = df[(df["Date"] >= START) & (df["Date"] <= END)]
    rep["rows_in_2012_2016"] = len(df)
    return df, rep


# ------------------------------------------------------------ aggregation
def monthly_series(df):
    """Aggregate to monthly demand per Warehouse x Category, zero-filled."""
    df = df.copy()
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    g = (df.groupby(["Warehouse", "Product_Category", "Month"])["Order_Demand"]
           .sum().reset_index())
    months = pd.date_range(START, END, freq="MS")
    frames = []
    for (w, c), part in g.groupby(["Warehouse", "Product_Category"]):
        s = part.set_index("Month")["Order_Demand"].reindex(months, fill_value=0)
        frames.append(pd.DataFrame({"Warehouse": w, "Product_Category": c,
                                    "Month": months, "Demand": s.values}))
    full = pd.concat(frames, ignore_index=True)
    stats = full.groupby(["Warehouse", "Product_Category"])["Demand"].agg(
        total="sum", active=lambda x: (x > 0).sum()).reset_index()
    keep = stats[(stats["active"] >= MIN_ACTIVE_MONTHS) &
                 (stats["total"] >= MIN_TOTAL_DEMAND)][["Warehouse", "Product_Category"]]
    full = full.merge(keep, on=["Warehouse", "Product_Category"])
    return full, stats


# -------------------------------------------------------------- features
def make_features(full):
    """Lag / rolling / calendar features. Everything uses data up to t-1 only."""
    d = full.sort_values(["Warehouse", "Product_Category", "Month"]).copy()
    key = ["Warehouse", "Product_Category"]
    g = d.groupby(key)["Demand"]
    for l in LAGS:
        d[f"lag_{l}"] = g.shift(l)
    shifted = g.shift(1)
    d["roll_mean_3"] = shifted.groupby([d[k] for k in key]).transform(lambda x: x.rolling(3).mean())
    d["roll_mean_6"] = shifted.groupby([d[k] for k in key]).transform(lambda x: x.rolling(6).mean())
    d["roll_std_3"] = shifted.groupby([d[k] for k in key]).transform(lambda x: x.rolling(3).std())
    d["month"] = d["Month"].dt.month
    d["quarter"] = d["Month"].dt.quarter
    d["year"] = d["Month"].dt.year
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["series_id"] = d["Warehouse"] + "|" + d["Product_Category"]
    d = d.dropna(subset=[f"lag_{l}" for l in LAGS] + ["roll_mean_6"]).reset_index(drop=True)
    return d


FEATURES = ([f"lag_{l}" for l in LAGS] +
            ["roll_mean_3", "roll_mean_6", "roll_std_3",
             "month", "quarter", "month_sin", "month_cos"])


def time_split(d, test_months=6):
    cutoff = d["Month"].max() - pd.DateOffset(months=test_months - 1)
    return d[d["Month"] < cutoff].copy(), d[d["Month"] >= cutoff].copy(), cutoff


# ------------------------------------------------------------- metrics
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    wape = np.abs(y - p).sum() / np.abs(y).sum() * 100      # robust to zero demand
    ss_res, ss_tot = ((y - p) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {"MAE": mae, "RMSE": rmse, "WAPE_%": wape, "R2": 1 - ss_res / ss_tot}


# ------------------------------------------------ model-ready matrices
LOG_COLS = [f"lag_{l}" for l in LAGS] + ["roll_mean_3", "roll_mean_6", "roll_std_3"]


def prep_X(d):
    """Log-transform demand-scale features so series of very different size can share one model."""
    X = d[FEATURES].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = (np.abs(y) + np.abs(p)) / 2
    return float(np.mean(np.where(den == 0, 0, np.abs(y - p) / np.where(den == 0, 1, den))) * 100)


def get_models():
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                  HistGradientBoostingRegressor)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Random Forest": RandomForestRegressor(300, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                random_state=42),
    }


def fit_predict(model, tr, te):
    model.fit(prep_X(tr), np.log1p(tr["Demand"].clip(lower=0)))
    return np.expm1(model.predict(prep_X(te))).clip(0)


# ---------------------------------------- recursive multi-step forecast
def forecast_future(model, full, warehouse, category, horizon=6):
    """Forecast `horizon` months beyond the last observed month for one series.
    Each predicted month is fed back as a lag for the next (recursive strategy)."""
    h = (full[(full.Warehouse == warehouse) & (full.Product_Category == category)]
         .sort_values("Month")[["Month", "Demand"]].reset_index(drop=True))
    out = []
    for _ in range(horizon):
        nxt = h["Month"].max() + pd.DateOffset(months=1)
        v = h["Demand"].values
        row = {f"lag_{l}": v[-l] for l in LAGS}
        row["roll_mean_3"], row["roll_mean_6"] = v[-3:].mean(), v[-6:].mean()
        row["roll_std_3"] = v[-3:].std(ddof=1)
        row.update(month=nxt.month, quarter=(nxt.month - 1) // 3 + 1,
                   month_sin=np.sin(2 * np.pi * nxt.month / 12),
                   month_cos=np.cos(2 * np.pi * nxt.month / 12))
        X = prep_X(pd.DataFrame([row]))
        pred = float(np.expm1(model.predict(X[FEATURES]))[0].clip(0))
        out.append({"Month": nxt, "Forecast": pred})
        h = pd.concat([h, pd.DataFrame({"Month": [nxt], "Demand": [pred]})], ignore_index=True)
    return pd.DataFrame(out)
