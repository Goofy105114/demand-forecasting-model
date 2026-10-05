"""
Production Demand Forecasting - Streamlit app
Run:  streamlit run app.py
Requires artifacts/ (create with: python train.py)
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import pipeline as P

st.set_page_config(page_title="Production Demand Forecasting", page_icon="🏭", layout="wide")


@st.cache_data
def load_tables():
    return {
        "monthly": pd.read_csv("artifacts/monthly_demand.csv", parse_dates=["Month"]),
        "compare": pd.read_csv("artifacts/model_comparison.csv", index_col=0),
        "cv": pd.read_csv("artifacts/cv_summary.csv", header=[0, 1], index_col=0),
        "per_series": pd.read_csv("artifacts/per_series_error.csv"),
        "test": pd.read_csv("artifacts/test_predictions.csv", parse_dates=["Month"]),
        "summary": json.load(open("artifacts/summary.json")),
    }


@st.cache_resource
def load_models():
    return joblib.load("artifacts/models.joblib")


try:
    T = load_tables()
    MODELS = load_models()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python train.py` first, then restart the app.")
    st.stop()

monthly, summary = T["monthly"], T["summary"]
champion = summary["champion"]

st.sidebar.title("🏭 Demand Forecasting")
page = st.sidebar.radio("Go to", ["Forecast", "Model comparison", "Data overview", "Error analysis", "About"])

# ----------------------------------------------------------------- Forecast
if page == "Forecast":
    st.title("Monthly demand forecast")
    st.caption("Pick a warehouse and product category, choose a model and horizon. "
               "Each forecast month feeds the next (recursive forecasting), so uncertainty grows with horizon.")
    c1, c2, c3, c4 = st.columns(4)
    wh = c1.selectbox("Warehouse", sorted(monthly["Warehouse"].unique()))
    cats = sorted(monthly[monthly["Warehouse"] == wh]["Product_Category"].unique())
    cat = c2.selectbox("Product category", cats)
    model_name = c3.selectbox("Model", list(MODELS.keys()),
                              index=list(MODELS.keys()).index(champion))
    horizon = c4.slider("Months ahead", 1, 6, 3)

    fc = P.forecast_future(MODELS[model_name], monthly, wh, cat, horizon)
    hist = monthly[(monthly["Warehouse"] == wh) & (monthly["Product_Category"] == cat)].sort_values("Month")

    # simple uncertainty band from the model's own hold-out error for this series
    ps = T["per_series"].set_index("series_id")
    sid = f"{wh}|{cat}"
    wape = float(ps.loc[sid, "WAPE_%"]) if sid in ps.index and pd.notna(ps.loc[sid, "WAPE_%"]) else 20.0
    band = wape / 100
    fc["Lower"] = fc["Forecast"] * (1 - band)
    fc["Upper"] = fc["Forecast"] * (1 + band)

    k1, k2, k3 = st.columns(3)
    k1.metric("Next-month forecast", f"{fc['Forecast'].iloc[0]:,.0f} units")
    k2.metric(f"Total next {horizon} month(s)", f"{fc['Forecast'].sum():,.0f} units")
    k3.metric("Hold-out error (WAPE) for this series", f"{wape:.1f}%")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist["Month"], hist["Demand"], label="History", color="#1f4e79")
    ax.plot(fc["Month"], fc["Forecast"], "o--", label="Forecast", color="#d9822b")
    ax.fill_between(fc["Month"], fc["Lower"], fc["Upper"], color="#d9822b", alpha=0.2,
                    label=f"±{wape:.0f}% (hold-out WAPE)")
    ax.set_ylabel("Units / month"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"{wh} · {cat} · {model_name}")
    st.pyplot(fig)

    out = fc.assign(Month=fc["Month"].dt.strftime("%Y-%m")).round(0)
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download forecast (CSV)", out.to_csv(index=False).encode(),
                       f"forecast_{wh}_{cat}.csv", "text/csv")

    st.subheader("Production planning view")
    safety = st.slider("Safety stock (% of forecast)", 0, 30, 10)
    stock = st.number_input("Current stock on hand (units)", min_value=0, value=0, step=1000)
    need = fc["Forecast"] * (1 + safety / 100)
    plan = pd.DataFrame({"Month": out["Month"], "Forecast": fc["Forecast"].round(0).values,
                         "Target incl. safety stock": need.round(0).values})
    cum = need.cumsum()
    plan["Production required"] = np.maximum(cum - stock, 0).diff().fillna(np.maximum(cum.iloc[0] - stock, 0)).round(0).values
    st.dataframe(plan, use_container_width=True, hide_index=True)

# ------------------------------------------------------ Model comparison
elif page == "Model comparison":
    st.title("Model comparison")
    st.write(f"Hold-out = last 6 months (from **{summary['test_start']}**), never seen in training. "
             f"Chosen model: **{champion}** (lowest mean error over 3 rolling-origin folds).")
    st.subheader("Hold-out results")
    st.dataframe(T["compare"].style.format("{:,.2f}"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    T["compare"]["WAPE_%"].sort_values().plot.barh(ax=ax, color="#1f4e79")
    ax.set_xlabel("WAPE % (lower is better)"); ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    st.subheader("Rolling-origin cross-validation (mean ± std over 3 folds)")
    cv = T["cv"].copy()
    cv.columns = [f"{a} {b}" for a, b in cv.columns]
    st.dataframe(cv, use_container_width=True)
    st.info("WAPE = total absolute error ÷ total actual demand. It is robust to zero-demand months, "
            "unlike MAPE. Baselines (naive, moving average) are included so we know ML is actually adding value.")

# ------------------------------------------------------- Data overview
elif page == "Data overview":
    st.title("Data overview")
    c = summary["cleaning"]
    cols = st.columns(4)
    cols[0].metric("Raw rows", f"{c['raw_rows']:,}")
    cols[1].metric("Missing dates dropped", f"{c['missing_dates_dropped']:,}")
    cols[2].metric("Bracketed (negative) values", f"{c['bracketed_negative_values']:,}")
    cols[3].metric("Series modelled", summary["n_series"])
    tot = monthly.groupby("Month")["Demand"].sum()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(tot.index, tot.values, color="#1f4e79"); ax.set_title("Total monthly demand (modelled series)")
    ax.grid(alpha=0.3); st.pyplot(fig)
    share = monthly.groupby("Product_Category")["Demand"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 3.8)); (share / 1e6).plot.bar(ax=ax, color="#d9822b")
    ax.set_ylabel("Total demand (millions)"); ax.set_title("Top 10 categories")
    st.pyplot(fig)
    st.dataframe(monthly.head(50), use_container_width=True)

# ------------------------------------------------------ Error analysis
elif page == "Error analysis":
    st.title("Error analysis (hold-out period)")
    ps = T["per_series"].copy()
    st.write("Error is very uneven across series. Big, steady series are predicted well; small, lumpy "
             "series are hard to forecast.")
    st.dataframe(ps.style.format({"Actual": "{:,.0f}", "AbsErr": "{:,.0f}", "WAPE_%": "{:.1f}"}),
                 use_container_width=True, hide_index=True)
    sid = st.selectbox("Inspect a series", ps["series_id"])
    w, cat = sid.split("|")
    t = T["test"][(T["test"]["Warehouse"] == w) & (T["test"]["Product_Category"] == cat)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t["Month"], t["Demand"], "o-", label="Actual"); ax.plot(t["Month"], t["Pred"], "s--", label="Predicted")
    ax.legend(); ax.grid(alpha=0.3); st.pyplot(fig)
    st.subheader("Known limitations")
    st.markdown("""
- Only ~5 years of monthly data, so yearly seasonality is weakly learned.
- Demand is *orders*, not true consumer demand; promotions, prices and stock-outs are not in the data.
- One category (Category_019) is ~83% of volume, so overall metrics mostly reflect it.
- Multi-step forecasts are recursive, so error compounds beyond ~3 months.
- Small / intermittent series (many zero months) have high percentage errors.
""")

# ------------------------------------------------------------- About
else:
    st.title("About this project")
    st.markdown(f"""
**Problem:** estimate future demand to support production planning for a manufacturer with four warehouses.

**Data:** *Forecasts for Product Demand* (Kaggle) – 1,048,575 order records, 2,160 products, 33 categories.

**Approach:** clean → aggregate to monthly demand per warehouse × category → lag / rolling / calendar features
→ time-based split → compare baselines and ML models → pick by rolling-origin CV → forecast with recursive prediction.

**Chosen model:** {champion}
""")
