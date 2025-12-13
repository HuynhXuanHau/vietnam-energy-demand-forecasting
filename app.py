import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
# TABLE STYLE (đậm hơn)
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
# SELECT COLS
# =========================
st.subheader("Chọn cột Time & Target")

col_time, col_target = st.columns([1, 1])

with col_time:
    time_col = st.selectbox(
        "Cột Time",
        raw_df.columns,
        index=0
    )

with col_target:
    target_col = st.selectbox(
        "Cột Target",
        raw_df.columns,
        index=min(1, len(raw_df.columns) - 1)
    )

df = raw_df[[time_col, target_col]].copy()
df.columns = ["Time_raw", "Target"]

# Target numeric
df["Target"] = pd.to_numeric(df["Target"], errors="coerce")

s = df["Time_raw"]

# Ưu tiên numeric trước để tránh Year bị hiểu nhầm thành epoch datetime
time_num = pd.to_numeric(s, errors="coerce")

# Nếu gần như toàn bộ là số và giá trị giống year (1000-3000) => coi là numeric (year)
is_mostly_numeric = time_num.notna().mean() >= 0.9
looks_like_year = is_mostly_numeric and time_num.between(1000, 3000).mean() >= 0.9

if looks_like_year:
    df["Time"] = time_num.astype(int)
    time_kind = "numeric"
else:
    # Nếu không phải year numeric thì mới thử parse datetime
    time_dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    if time_dt.notna().mean() >= 0.6 and time_dt.nunique() >= 3:
        df["Time"] = time_dt
        time_kind = "datetime"
    elif is_mostly_numeric:
        df["Time"] = time_num.astype(float)
        time_kind = "numeric"
    else:
        df["Time"] = s.astype(str)
        time_kind = "string"

df = df.dropna(subset=["Target"]).copy()

# sort by Time nếu sort được
if time_kind in ("datetime", "numeric"):
    df = df.sort_values("Time").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# Index nội bộ cho model
df["Index"] = np.arange(len(df))

st.subheader("Dữ liệu sau xử lý")
st.dataframe(style_table(df[["Time", "Target"]]), use_container_width=True)

# =========================
# FUTURE TIME GENERATOR
# =========================
def make_future_time(series_time: pd.Series, steps: int, kind: str) -> pd.Series:
    if steps <= 0:
        return pd.Series([], dtype=object)

    if kind == "datetime":
        t = pd.to_datetime(series_time, errors="coerce")
        t = t.dropna()
        if len(t) >= 2:
            delta = t.iloc[-1] - t.iloc[-2]
            # nếu delta = 0 (trùng ngày), fallback 1 day
            if delta == pd.Timedelta(0):
                delta = pd.Timedelta(days=1)
        else:
            delta = pd.Timedelta(days=1)
        start = pd.to_datetime(series_time.iloc[-1])
        return pd.Series([start + delta * (i + 1) for i in range(steps)])

    if kind == "numeric":
        t = pd.to_numeric(series_time, errors="coerce")
        t = t.dropna()
        if len(t) >= 2:
            delta = t.iloc[-1] - t.iloc[-2]
            if delta == 0:
                delta = 1
        else:
            delta = 1
        last = float(series_time.iloc[-1])
        return pd.Series([last + delta * (i + 1) for i in range(steps)])

    # string fallback: hiển thị t+1, t+2...
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

    # làm thanh ngắn: đặt trong cột hẹp hơn
    c1, c2 = st.columns([1, 1])
    with c1:
        test_pct = st.slider("Test (%)", 10, 40, 20, step=5)
    with c2:
        forecast_steps = st.number_input("Forecast steps", 1, 60, 6, step=1)

    # split
    n = len(df)
    n_test = max(1, int(n * test_pct / 100))
    n_train = n - n_test

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    st.markdown("### Split")
    st.write(f"Train: **{n_train}** mẫu")
    st.write(f"Test: **{n_test}** mẫu")

    # =========================
    # TRAIN / PREDICT
    # =========================
    x_train = train_df["Index"].to_numpy()
    x_test = test_df["Index"].to_numpy()

    y_train = train_df["Target"].to_numpy()
    y_test = test_df["Target"].to_numpy()

    future_index = np.arange(df["Index"].iloc[-1] + 1, df["Index"].iloc[-1] + 1 + forecast_steps)
    meta = {}

    with st.spinner("Đang train & dự báo..."):
        if model_name.startswith("Holt"):
            m = fit_holt_additive(y_train)
            train_pred = np.asarray(m.fittedvalues, dtype=float)
            test_pred = np.asarray(predict_holt(m, steps=len(test_df)), dtype=float)

            m_full = fit_holt_additive(df["Target"].to_numpy())
            future_pred = np.asarray(predict_holt(m_full, steps=forecast_steps), dtype=float)

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
            pred_future_full = gm11_lambda_predict(df["Target"].to_numpy(), best_lambda_full, n_forecast=forecast_steps)
            future_pred = np.asarray(pred_future_full[-forecast_steps:], dtype=float)
            meta["best_lambda"] = float(best_lambda)

    # =========================
    # METRICS AS TEXT (không JSON)
    # =========================
    st.markdown("### Metrics (Train)")
    tr = compute_metrics(y_train, train_pred)
    st.write(f"- **MAE**: {tr['MAE']:.4f}")
    st.write(f"- **RMSE**: {tr['RMSE']:.4f}")
    st.write(f"- **MAPE(%)**: {tr['MAPE(%)']:.4f}")

    st.markdown("### Metrics (Test)")
    te = compute_metrics(y_test, test_pred)
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

    def set_time_ticks(ax, time_values, max_ticks=8):
        """
        Hiển thị giá trị Time rõ ràng trên trục X
        """
        time_series = pd.Series(time_values).dropna()
        n = len(time_values)
        if n <= max_ticks:
            ticks = time_values
        else:
            idx = np.linspace(0, n - 1, max_ticks, dtype=int)
            ticks = time_values.iloc[idx]

        ax.set_xticks(ticks)

        # Nếu là datetime → format đẹp
        if np.issubdtype(time_values.dtype, np.datetime64):
            labels = [t.strftime("%Y-%m-%d") for t in ticks]
        else:
            if np.issubdtype(time_series.dtype, np.datetime64):
                labels = [t.strftime("%Y") for t in ticks]

            elif np.issubdtype(time_series.dtype, np.number):
                # BỎ .0
                labels = [str(int(round(t))) for t in ticks]

            else:
                labels = [str(t) for t in ticks]

        ax.set_xticklabels(labels, rotation=30, ha="right")

    def plot_pretty_train_test():
        fig = plt.figure(figsize=(11, 4.6))
        ax = plt.gca()

        # Actual/Pred Train
        ax.plot(train_df["Time"], y_train, marker="o", linewidth=2.2, label="Actual (Train)")
        ax.plot(train_df["Time"], train_pred, marker=".", linestyle="--", linewidth=2.0, label="Pred (Train)")

        # Actual/Pred Test
        ax.plot(test_df["Time"], y_test, marker="o", linewidth=2.2, label="Actual (Test)")
        ax.plot(test_df["Time"], test_pred, marker=".", linestyle="--", linewidth=2.0, label="Pred (Test)")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        set_time_ticks(ax, df["Time"])
        plt.tight_layout()
        return fig

    def plot_pretty_forecast():
        future_time = make_future_time(df["Time"], forecast_steps, time_kind)

        fig = plt.figure(figsize=(11, 4.6))
        ax = plt.gca()

        ax.plot(df["Time"], df["Target"], marker="o", linewidth=2.2, label="Actual")
        ax.plot(future_time, future_pred, marker="*", linestyle="--", linewidth=2.0, label=f"Forecast (+{forecast_steps})")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        all_time = pd.concat([df["Time"], future_time], ignore_index=True)
        set_time_ticks(ax, all_time)

        plt.tight_layout()
        return fig, future_time

    st.markdown("### Train/Test")
    fig1 = plot_pretty_train_test()
    st.pyplot(fig1, clear_figure=True)

    st.markdown("### Forecast")
    fig2, future_time = plot_pretty_forecast()
    st.pyplot(fig2, clear_figure=True)

# =========================
# FORECAST TABLE (Time thật)
# =========================
out = pd.DataFrame({
    "Time": future_time,
    "Forecast": future_pred
})

# ----- FORMAT HIỂN THỊ -----

# Nếu Time là numeric (year, step, …) → bỏ .000
if np.issubdtype(out["Time"].dtype, np.number):
    out["Time"] = out["Time"].astype(int)

# Nếu Time là datetime → format đẹp
elif np.issubdtype(out["Time"].dtype, np.datetime64):
    out["Time"] = out["Time"].dt.strftime("%Y-%m-%d")

# Làm tròn forecast 2 chữ số
out["Forecast"] = out["Forecast"].round(2)

st.dataframe(style_table(out), use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Tải forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")
