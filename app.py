import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from models import (
    compute_metrics,
    fit_linear, predict_linear,
    fit_poly_with_tscv, predict_poly,
    fit_holt_additive, predict_holt,
    fit_svr_rbf, predict_svr,
    fit_pso_gm11, gm11_lambda_predict
)

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Generic Time-Series Forecasting", layout="wide")
st.title("Time-Series Forecasting App")

# =========================
# TABLE STYLE
# =========================
def style_table(df: pd.DataFrame):
    return (
        df.style
        .set_properties(**{
            "font-size": "15px",
            "font-weight": "650",
            "border": "1.6px solid #555",
            "padding": "8px"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("font-size", "16px"),
                ("font-weight", "900"),
                ("background-color", "#f0f2f6"),
                ("border", "2px solid #444"),
                ("padding", "10px"),
                ("text-align", "left")
            ]},
            {"selector": "td", "props": [
                ("border", "1.3px solid #666"),
                ("padding", "8px")
            ]},
            {"selector": "table", "props": [("border-collapse", "collapse")]}
        ])
    )

# =========================
# ROBUST PARSERS (TỔNG QUÁT)
# =========================
def parse_numeric_series(s: pd.Series) -> pd.Series:
    """
    Parse numeric values robustly:
    - 1,234,567
    - 1.234.567
    - 1,234.56
    - 1.234,56
    - '2,054,309 khách'
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+", "", regex=True)

    # keep digits, separators, minus
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)

    mask_both = s.str.contains(",") & s.str.contains(".")

    def normalize(x: str) -> str:
        if "," in x and "." in x:
            # if last separator is "," => EU decimal
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        else:
            # only one type of sep:
            # heuristic: if contains ',' and has <=2 digits after it => decimal
            if "," in x:
                if len(x.split(",")[-1]) <= 2:
                    x = x.replace(".", "").replace(",", ".")
                else:
                    x = x.replace(",", "")
            elif "." in x:
                if len(x.split(".")[-1]) <= 2:
                    x = x.replace(",", "")
                else:
                    x = x.replace(".", "")
        return x

    s.loc[mask_both] = s.loc[mask_both].apply(normalize)
    s.loc[~mask_both] = s.loc[~mask_both].apply(normalize)

    return pd.to_numeric(s, errors="coerce")


def parse_time_column(s: pd.Series):
    """
    Return (parsed_series, time_kind) where time_kind in: numeric | datetime | string
    Supports:
    - Year: 2020, 2020.0
    - Month-year: Jan-22, Mar-25, 2022-03, 03-2022, 03/2022...
    - Date formats
    """
    s_raw = s.copy()

    # 1) numeric first to avoid Year -> 1970 datetime bug
    time_num = pd.to_numeric(s, errors="coerce")
    is_mostly_numeric = time_num.notna().mean() >= 0.9
    looks_like_year = is_mostly_numeric and time_num.between(1000, 3000).mean() >= 0.9
    if looks_like_year:
        return time_num.astype(int), "numeric"

    # 2) try datetime auto
    time_dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # 3) try common formats if fail a lot
    if time_dt.notna().mean() < 0.6:
        formats = ["%b-%y", "%b-%Y", "%Y-%m", "%Y/%m", "%m-%Y", "%m/%Y", "%Y"]
        best = time_dt
        for f in formats:
            tmp = pd.to_datetime(s.astype(str).str.strip(), format=f, errors="coerce")
            if tmp.notna().mean() > best.notna().mean():
                best = tmp
        time_dt = best

    if time_dt.notna().mean() >= 0.6 and time_dt.nunique() >= 3:
        return time_dt, "datetime"

    # 4) numeric but not year
    if is_mostly_numeric:
        return time_num.astype(float), "numeric"

    # 5) fallback string
    return s_raw.astype(str), "string"


# =========================
# LOAD RAW
# =========================
@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df

uploaded = st.file_uploader("Upload dataset (.xlsx / .csv)", type=["xlsx", "csv"])
if uploaded is None:
    st.info("Upload file để bắt đầu.")
    st.stop()

raw_df = load_raw(uploaded.getvalue(), uploaded.name)

st.subheader("Dữ liệu gốc")
st.dataframe(style_table(raw_df), use_container_width=True, height=260)

# =========================
# SELECT COLS (1 HÀNG)
# =========================
st.subheader("Chọn cột Time & Target")
col_time, col_target = st.columns([1, 1])

with col_time:
    time_col = st.selectbox("Cột Time", raw_df.columns, index=0)

with col_target:
    target_col = st.selectbox("Cột Target", raw_df.columns, index=min(1, len(raw_df.columns) - 1))

df = raw_df[[time_col, target_col]].copy()
df.columns = ["Time_raw", "Target_raw"]

# Parse Target robustly
df["Target"] = parse_numeric_series(df["Target_raw"])

# Parse Time robustly
df["Time"], time_kind = parse_time_column(df["Time_raw"])

# Drop invalid target
df = df.dropna(subset=["Target"]).copy()

# Stop if empty / too small
if df.empty or len(df) < 3:
    st.error(
        "Dữ liệu sau xử lý không đủ (hoặc Target không parse được).\n"
        "Hãy kiểm tra lại cột Target (dấu phẩy/chấm/ký tự) và cột Time."
    )
    st.stop()

# Sort
if time_kind in ("datetime", "numeric"):
    df = df.sort_values("Time").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# Internal index for models
df["Index"] = np.arange(len(df))

st.subheader("Dữ liệu sau xử lý")
show_df = df[["Time", "Target"]].copy()

# display nicer Target
show_df["Target"] = show_df["Target"].round(2)

# display nicer Time
if time_kind == "numeric":
    # remove .0
    show_df["Time"] = pd.to_numeric(show_df["Time"], errors="coerce").astype("Int64")
elif time_kind == "datetime":
    inferred = pd.infer_freq(pd.to_datetime(df["Time"].dropna()))
    if inferred and "M" in inferred:
        show_df["Time"] = pd.to_datetime(show_df["Time"]).dt.strftime("%b-%y")
    elif inferred and ("A" in inferred or "Y" in inferred):
        show_df["Time"] = pd.to_datetime(show_df["Time"]).dt.strftime("%Y")
    else:
        show_df["Time"] = pd.to_datetime(show_df["Time"]).dt.strftime("%Y-%m-%d")

st.dataframe(style_table(show_df), use_container_width=True)

# =========================
# FUTURE TIME GENERATOR
# =========================
def make_future_time(series_time: pd.Series, steps: int, kind: str) -> pd.Series:
    if steps <= 0:
        return pd.Series([], dtype=object)

    if kind == "datetime":
        t = pd.to_datetime(series_time, errors="coerce").dropna()
        if len(t) == 0:
            return pd.Series([pd.NaT] * steps)

        inferred = pd.infer_freq(t)
        last = t.iloc[-1]

        if inferred is not None:
            rng = pd.date_range(start=last, periods=steps + 1, freq=inferred)[1:]
            return pd.Series(rng)

        if len(t) >= 2:
            delta = t.iloc[-1] - t.iloc[-2]
            if delta == pd.Timedelta(0):
                delta = pd.Timedelta(days=1)
            return pd.Series([last + delta * (i + 1) for i in range(steps)])

        return pd.Series([last + pd.Timedelta(days=i + 1) for i in range(steps)])

    if kind == "numeric":
        t = pd.to_numeric(series_time, errors="coerce").dropna()
        if len(t) >= 2:
            delta = t.iloc[-1] - t.iloc[-2]
            if delta == 0:
                delta = 1
        else:
            delta = 1
        last = float(pd.to_numeric(series_time.iloc[-1], errors="coerce"))
        return pd.Series([last + delta * (i + 1) for i in range(steps)])

    return pd.Series([f"t+{i+1}" for i in range(steps)])

# =========================
# LAYOUT
# =========================
left, right = st.columns([0.75, 2.25])

with left:
    st.markdown("### Cấu hình")
    model_name = st.selectbox(
        "Chọn mô hình",
        ["Holt (Additive)", "SVR (RBF)", "Polynomial Regression", "Linear Regression", "PSO-GM(1,1)"],
        index=0
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        test_pct = st.slider("Test (%)", 10, 40, 20, step=5)
    with c2:
        forecast_steps = st.number_input("Forecast steps", 1, 60, 6, step=1)

    n = len(df)
    n_test = max(1, int(n * test_pct / 100))
    n_train = max(1, n - n_test)

    # ensure at least 2 points for training some models
    if n_train < 2:
        st.error("Train quá ít. Giảm % test hoặc dùng dữ liệu dài hơn.")
        st.stop()

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    st.markdown("### Split")
    st.write(f"Train: **{len(train_df)}** mẫu")
    st.write(f"Test: **{len(test_df)}** mẫu")

    x_train = train_df["Index"].to_numpy()
    x_test = test_df["Index"].to_numpy()

    y_train = train_df["Target"].to_numpy()
    y_test = test_df["Target"].to_numpy()

    # safe future_index
    last_idx = int(df["Index"].iloc[-1])
    future_index = np.arange(last_idx + 1, last_idx + 1 + int(forecast_steps))

    meta = {}

    with st.spinner("Đang train & dự báo..."):
        if model_name.startswith("Holt"):
            m = fit_holt_additive(y_train)
            train_pred = np.asarray(m.fittedvalues, dtype=float)
            test_pred = np.asarray(predict_holt(m, steps=len(test_df)), dtype=float)

            m_full = fit_holt_additive(df["Target"].to_numpy())
            future_pred = np.asarray(predict_holt(m_full, steps=int(forecast_steps)), dtype=float)

        elif model_name.startswith("SVR"):
            tmp_train = train_df.rename(columns={"Index": "Year", "Target": "Total_National_Demand"})
            model, meta = fit_svr_rbf(tmp_train)
            train_pred = np.asarray(predict_svr(model, x_train), dtype=float)
            test_pred = np.asarray(predict_svr(model, x_test), dtype=float)

            tmp_full = df.rename(columns={"Index": "Year", "Target": "Total_National_Demand"})
            model_full, _ = fit_svr_rbf(tmp_full)
            future_pred = np.asarray(predict_svr(model_full, future_index), dtype=float)

        elif model_name.startswith("Polynomial"):
            tmp_train = train_df.rename(columns={"Index": "Year", "Target": "Total_National_Demand"})
            poly, meta = fit_poly_with_tscv(tmp_train)
            train_pred = np.asarray(predict_poly(poly, x_train), dtype=float)
            test_pred = np.asarray(predict_poly(poly, x_test), dtype=float)

            tmp_full = df.rename(columns={"Index": "Year", "Target": "Total_National_Demand"})
            poly_full, _ = fit_poly_with_tscv(tmp_full)
            future_pred = np.asarray(predict_poly(poly_full, future_index), dtype=float)

        elif model_name.startswith("Linear"):
            tmp_train = train_df.rename(columns={"Index": "Year", "Target": "Total_National_Demand"})
            lin = fit_linear(tmp_train)
            train_pred = np.asarray(predict_linear(lin, x_train), dtype=float)
            test_pred = np.asarray(predict_linear(lin, x_test), dtype=float)

            tmp_full = df.rename(columns={"Index": "Year", "Target": "Total_National_Demand"})
            lin_full = fit_linear(tmp_full)
            future_pred = np.asarray(predict_linear(lin_full, future_index), dtype=float)

        else:
            best_lambda, meta = fit_pso_gm11(y_train)
            pred_train_full = gm11_lambda_predict(y_train, best_lambda, n_forecast=0)
            train_pred = np.asarray(pred_train_full[:len(train_df)], dtype=float)

            pred_test_full = gm11_lambda_predict(y_train, best_lambda, n_forecast=len(test_df))
            test_pred = np.asarray(pred_test_full[-len(test_df):], dtype=float)

            best_lambda_full, _ = fit_pso_gm11(df["Target"].to_numpy())
            pred_future_full = gm11_lambda_predict(df["Target"].to_numpy(), best_lambda_full, n_forecast=int(forecast_steps))
            future_pred = np.asarray(pred_future_full[-int(forecast_steps):], dtype=float)
            meta["best_lambda"] = float(best_lambda)

    # METRICS
    st.markdown("### Metrics (Train)")
    tr = compute_metrics(y_train, train_pred)
    st.write(f"- **MAE**: {tr['MAE']:.4f}")
    st.write(f"- **RMSE**: {tr['RMSE']:.4f}")
    st.write(f"- **MAPE(%)**: {tr['MAPE(%)']:.4f}")

    st.markdown("### Metrics (Test)")
    te = compute_metrics(y_test, test_pred) if len(test_df) > 0 else {"MAE": np.nan, "RMSE": np.nan, "MAPE(%)": np.nan}
    st.write(f"- **MAE**: {te['MAE']:.4f}")
    st.write(f"- **RMSE**: {te['RMSE']:.4f}")
    st.write(f"- **MAPE(%)**: {te['MAPE(%)']:.4f}")

    if meta:
        st.markdown("### Thông tin thêm")
        for k, v in meta.items():
            if isinstance(v, dict):
                st.write(f"**{k}:**")
                for kk, vv in v.items():
                    st.write(f"- {kk}: {vv}")
            else:
                st.write(f"- **{k}**: {v}")

with right:
    st.markdown("## Biểu đồ")

    def set_time_ticks(ax, time_values, max_ticks=10):
        s = pd.Series(time_values).dropna()
        if len(s) == 0:
            return

        n = len(s)
        idx = np.arange(n) if n <= max_ticks else np.linspace(0, n - 1, max_ticks, dtype=int)
        ticks = s.iloc[idx]

        ax.set_xticks(ticks)

        if is_datetime64_any_dtype(s):
            inferred = pd.infer_freq(pd.to_datetime(s))
            fmt = "%b-%y" if inferred and "M" in inferred else "%Y-%m-%d"
            labels = [pd.to_datetime(t).strftime(fmt) for t in ticks]
        elif is_numeric_dtype(s):
            labels = [str(int(round(float(t)))) for t in ticks]  # bỏ .0
        else:
            labels = [str(t) for t in ticks]

        ax.set_xticklabels(labels, rotation=30, ha="right")

    def plot_pretty_train_test():
        fig = plt.figure(figsize=(11, 4.6))
        ax = plt.gca()

        ax.plot(train_df["Time"], y_train, marker="o", linewidth=2.2, label="Actual (Train)")
        ax.plot(train_df["Time"], train_pred, marker=".", linestyle="--", linewidth=2.0, label="Pred (Train)")

        if len(test_df) > 0:
            ax.plot(test_df["Time"], y_test, marker="o", linewidth=2.2, label="Actual (Test)")
            ax.plot(test_df["Time"], test_pred, marker=".", linestyle="--", linewidth=2.0, label="Pred (Test)")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)

        set_time_ticks(ax, df["Time"], max_ticks=12)
        plt.tight_layout()
        return fig

    def plot_pretty_forecast():
        future_time = make_future_time(df["Time"], int(forecast_steps), time_kind)

        fig = plt.figure(figsize=(11, 4.6))
        ax = plt.gca()

        ax.plot(df["Time"], df["Target"], marker="o", linewidth=2.2, label="Actual")
        ax.plot(future_time, future_pred, marker="*", linestyle="--", linewidth=2.0, label=f"Forecast (+{int(forecast_steps)})")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)

        all_time = pd.concat([df["Time"], future_time], ignore_index=True)
        set_time_ticks(ax, all_time, max_ticks=12)

        plt.tight_layout()
        return fig, future_time

    st.markdown("### Train/Test")
    st.pyplot(plot_pretty_train_test(), clear_figure=True)

    st.markdown("### Forecast")
    fig2, future_time = plot_pretty_forecast()
    st.pyplot(fig2, clear_figure=True)

# =========================
# FORECAST TABLE
# =========================
out = pd.DataFrame({
    "Time": future_time,
    "Forecast": future_pred
})

# format Time
if is_numeric_dtype(out["Time"]):
    out["Time"] = pd.to_numeric(out["Time"], errors="coerce").round().astype("Int64")
elif is_datetime64_any_dtype(out["Time"]):
    inferred = pd.infer_freq(pd.to_datetime(df["Time"].dropna()))
    if inferred and "M" in inferred:
        out["Time"] = pd.to_datetime(out["Time"]).dt.strftime("%b-%y")
    elif inferred and ("A" in inferred or "Y" in inferred):
        out["Time"] = pd.to_datetime(out["Time"]).dt.strftime("%Y")
    else:
        out["Time"] = pd.to_datetime(out["Time"]).dt.strftime("%Y-%m-%d")

# format Forecast
out["Forecast"] = pd.to_numeric(out["Forecast"], errors="coerce").round(2)

st.markdown("### Bảng dự báo")
st.dataframe(style_table(out), use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Tải forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")
