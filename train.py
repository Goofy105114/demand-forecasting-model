"""Train, evaluate and save everything the Streamlit app needs.  Run:  python train.py"""
import json, os, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import pipeline as P

os.makedirs("artifacts", exist_ok=True)
raw = P.load_raw()
df, report = P.clean(raw)
full, _ = P.monthly_series(df)
d = P.make_features(full)
tr, te, cutoff = P.time_split(d, 6)
print("cleaning report:", report)
print(f"series={d.series_id.nunique()}  train={len(tr)}  test={len(te)}  test starts {cutoff:%Y-%m}")

# ---- 1. baselines + models on the hold-out (last 6 months) -------------
preds = {"Naive (last month)": te["lag_1"].values,
         "Seasonal naive (12 mo ago)": te["lag_12"].values,
         "Moving average (3 mo)": te["roll_mean_3"].values}
models = P.get_models()
for name, m in models.items():
    preds[name] = P.fit_predict(m, tr, te)
rows = []
for name, p in preds.items():
    r = P.metrics(te["Demand"], p); r["sMAPE_%"] = P.smape(te["Demand"], p); r["Model"] = name
    rows.append(r)
res = pd.DataFrame(rows).set_index("Model")[["MAE", "RMSE", "WAPE_%", "sMAPE_%", "R2"]].round(3)
print(res)

# ---- 2. rolling-origin cross-validation (3 folds x 6 months) -----------
cv_rows = []
months = sorted(d["Month"].unique())
for k, cut in enumerate([months[-18], months[-12], months[-6]], 1):
    a = d[d["Month"] < cut]
    b = d[(d["Month"] >= cut) & (d["Month"] < cut + pd.DateOffset(months=6))]
    base = {"Moving average (3 mo)": b["roll_mean_3"].values}
    for name, m in P.get_models().items():
        base[name] = P.fit_predict(m, a, b)
    for name, p in base.items():
        cv_rows.append({"Fold": k, "Model": name, "WAPE_%": P.metrics(b["Demand"], p)["WAPE_%"],
                        "sMAPE_%": P.smape(b["Demand"], p)})
cv = pd.DataFrame(cv_rows)
cv_summary = cv.groupby("Model")[["WAPE_%", "sMAPE_%"]].agg(["mean", "std"]).round(2)
print(cv_summary)

# ---- 3. light hyper-parameter search for the best family ---------------
from sklearn.ensemble import HistGradientBoostingRegressor
grid = [(lr, it, leaf) for lr in (0.03, 0.05, 0.1) for it in (200, 400) for leaf in (10, 20)]
tune = []
inner_cut = months[-12]
a = d[d["Month"] < inner_cut]; b = d[(d["Month"] >= inner_cut) & (d["Month"] < cutoff)]
for lr, it, leaf in grid:
    m = HistGradientBoostingRegressor(learning_rate=lr, max_iter=it, min_samples_leaf=leaf, random_state=42)
    p = P.fit_predict(m, a, b)
    tune.append({"learning_rate": lr, "max_iter": it, "min_samples_leaf": leaf,
                 "val_WAPE_%": P.metrics(b["Demand"], p)["WAPE_%"]})
tune = pd.DataFrame(tune).sort_values("val_WAPE_%").reset_index(drop=True)
best = tune.iloc[0]; print(tune.head(5))
tuned = HistGradientBoostingRegressor(learning_rate=float(best.learning_rate), max_iter=int(best.max_iter),
                                      min_samples_leaf=int(best.min_samples_leaf), random_state=42)
p_t = P.fit_predict(tuned, tr, te)
r = P.metrics(te["Demand"], p_t); r["sMAPE_%"] = P.smape(te["Demand"], p_t)
res.loc["Hist Gradient Boosting (tuned)"] = pd.Series(r)[res.columns].round(3)
preds["Hist Gradient Boosting (tuned)"] = p_t

# ---- 4. final model choice & error analysis ----------------------------
# Champion = lowest MEAN WAPE over the 3 rolling-origin folds (more robust than one window).
ml_cv = cv_summary[("WAPE_%", "mean")].drop(index=["Moving average (3 mo)"])
champion = ml_cv.idxmin()
print("holdout table incl. tuned HGB:"); print(res)
print("champion:", champion)
te = te.assign(Pred=preds[champion])
te["AbsErr"] = (te["Demand"] - te["Pred"]).abs()
per_series = (te.groupby("series_id").agg(Actual=("Demand", "sum"), AbsErr=("AbsErr", "sum")).reset_index())
per_series["WAPE_%"] = per_series["AbsErr"] / per_series["Actual"].replace(0, np.nan) * 100
per_series = per_series.sort_values("Actual", ascending=False)

# ---- 5. refit champion on ALL data for future forecasts ----------------
final_models = {"Hist Gradient Boosting (tuned)": tuned, **P.get_models()}
final = {}
for name, m in final_models.items():
    m.fit(P.prep_X(d), np.log1p(d["Demand"].clip(lower=0)))
    final[name] = m
joblib.dump(final, "artifacts/models.joblib")
joblib.dump(final[champion], "artifacts/champion.joblib")

full.to_csv("artifacts/monthly_demand.csv", index=False)
res.to_csv("artifacts/model_comparison.csv")
cv_summary.to_csv("artifacts/cv_summary.csv")
tune.to_csv("artifacts/tuning.csv", index=False)
per_series.to_csv("artifacts/per_series_error.csv", index=False)
te[["Warehouse", "Product_Category", "Month", "Demand", "Pred"]].to_csv("artifacts/test_predictions.csv", index=False)
json.dump({"cleaning": report, "champion": champion, "test_start": str(cutoff.date()),
           "n_series": int(d.series_id.nunique()), "best_params": best.to_dict()},
          open("artifacts/summary.json", "w"), indent=2, default=str)
print("saved artifacts/")
