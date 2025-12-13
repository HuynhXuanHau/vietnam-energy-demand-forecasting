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

st.set_page_config(page_title="Energy Demand Forecasting", layout="wide")

st.title(" Vietnam Energy Demand Forecasting")

with st.expander("Hướng dẫn nhanh", expanded=True):
    st.markdown(
        """
- Upload file **Excel/CSV** có 2 cột: `Year`, `Total_National_Demand` (đơn vị TWh).
- App sẽ tự split: **Train 1986–2016**, **Test 2017–2024**.
- Chọn model → xem metrics + biểu đồ + dự báo 2025–2030.
        """
    )

uploaded = st.file_uploader("Upload dataset (.xlsx/.csv)", type=["xlsx", "csv"])

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    # normalize cols
    df.columns = [c.strip() for c in df.columns]
    if "Year" not in df.columns or "Total_National_Demand" not in df.columns:
        raise ValueError("File must contain columns: Year, Total_National_Demand")
    df = df[["Year", "Total_National_Demand"]].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Total_National_Demand"] = pd.to_numeric(df["Total_National_Demand"], errors="coerce")
    df = df.dropna().sort_values("Year")
    return df

if uploaded is None:
    st.info("Chưa có dữ liệu. Bạn upload file để bắt đầu.")
    st.stop()

try:
    data_df = load_data(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"Lỗi đọc file: {e}")
    st.stop()

st.subheader("Dữ liệu")
st.dataframe(data_df, use_container_width=True, height=260)

train_df = data_df[data_df["Year"] <= 2016].copy()
test_df = data_df[data_df["Year"] >= 2017].copy()

if len(train_df) < 10 or len(test_df) < 3:
    st.warning("Dữ liệu có vẻ thiếu (train/test quá ít). Hãy kiểm tra lại năm 1986–2024.")

colA, colB, colC = st.columns([1.2, 1.2, 2.2])

with colA:
    model_name = st.radio(
        "Chọn mô hình",
        ["Holt (Additive)", "SVR (RBF)", "Polynomial Regression", "Linear Regression", "PSO-GM(1,1) (optional)"],
        index=0
    )

with colB:
    forecast_start = st.number_input("Forecast start year", value=2025, min_value=1900, max_value=2100, step=1)
    forecast_end = st.number_input("Forecast end year", value=2030, min_value=1900, max_value=2100, step=1)
    if forecast_end < forecast_start:
        st.error("Forecast end year phải >= start year.")
        st.stop()
    future_years = np.arange(int(forecast_start), int(forecast_end) + 1)

with colC:
    st.markdown("### Split")
    st.write(f"Train: **{int(train_df['Year'].min())}–{int(train_df['Year'].max())}** (n={len(train_df)})")
    st.write(f"Test: **{int(test_df['Year'].min())}–{int(test_df['Year'].max())}** (n={len(test_df)})")

# -------------------------
# Train + predict
# -------------------------
train_years = train_df["Year"].to_numpy()
test_years = test_df["Year"].to_numpy()

y_train = train_df["Total_National_Demand"].to_numpy()
y_test = test_df["Total_National_Demand"].to_numpy()

train_pred = None
test_pred = None
future_pred = None
meta = {}

with st.spinner("Đang train & dự báo..."):
    if model_name.startswith("Holt"):
        m = fit_holt_additive(y_train)
        train_pred = np.asarray(m.fittedvalues, dtype=float)
        test_pred = predict_holt(m, steps=len(test_df))
        # refit on full for future
        m_full = fit_holt_additive(data_df["Total_National_Demand"].to_numpy())
        future_pred = predict_holt(m_full, steps=len(future_years))
    elif model_name.startswith("SVR"):
        bundle, meta = fit_svr_rbf(train_df)
        train_pred = predict_svr(bundle, train_years)
        test_pred = predict_svr(bundle, test_years)
        # refit on full
        bundle_full, _ = fit_svr_rbf(data_df)
        future_pred = predict_svr(bundle_full, future_years)
    elif model_name.startswith("Polynomial"):
        poly_model, meta = fit_poly_with_tscv(train_df)
        train_pred = predict_poly(poly_model, train_years)
        test_pred = predict_poly(poly_model, test_years)
        # refit on full with same degree isn't preserved; for demo refit again:
        poly_full, _ = fit_poly_with_tscv(data_df)
        future_pred = predict_poly(poly_full, future_years)
    elif model_name.startswith("Linear"):
        lin = fit_linear(train_df)
        train_pred = predict_linear(lin, train_years)
        test_pred = predict_linear(lin, test_years)
        lin_full = fit_linear(data_df)
        future_pred = predict_linear(lin_full, future_years)
    else:
        # PSO-GM(1,1)
        try:
            best_lambda, meta = fit_pso_gm11(y_train)
            pred_full_test = gm11_lambda_predict(y_train, best_lambda, n_forecast=len(test_df))
            test_pred = pred_full_test[-len(test_df):]
            # "train_pred" for GM is approximate: fitted on train itself
            pred_full_train = gm11_lambda_predict(y_train, best_lambda, n_forecast=0)
            train_pred = pred_full_train[:len(train_df)]
            # fit on full for future
            best_lambda_full, _ = fit_pso_gm11(data_df["Total_National_Demand"].to_numpy())
            pred_full_future = gm11_lambda_predict(data_df["Total_National_Demand"].to_numpy(), best_lambda_full, n_forecast=len(future_years))
            future_pred = pred_full_future[-len(future_years):]
            meta["best_lambda"] = float(best_lambda)
        except Exception as e:
            st.error(str(e))
            st.stop()

# -------------------------
# Metrics
# -------------------------
train_metrics = compute_metrics(y_train, train_pred)
test_metrics = compute_metrics(y_test, test_pred)

m1, m2 = st.columns(2)
with m1:
    st.markdown("### Metrics (Train)")
    st.json(train_metrics)
with m2:
    st.markdown("### Metrics (Test)")
    st.json(test_metrics)

if meta:
    st.markdown("### Thông tin thêm")
    st.json(meta)

# -------------------------
# Plots
# -------------------------
st.markdown("### Biểu đồ")
fig = plt.figure(figsize=(10, 4.2))
plt.plot(train_years, y_train, marker="o", linestyle="-", label="Actual (Train)")
plt.plot(train_years, train_pred, marker=".", linestyle="--", label="Pred (Train)")
plt.plot(test_years, y_test, marker="o", linestyle="-", label="Actual (Test)")
plt.plot(test_years, test_pred, marker=".", linestyle="--", label="Pred (Test)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Total_National_Demand (TWh)")
plt.legend()
st.pyplot(fig, clear_figure=True)

fig2 = plt.figure(figsize=(10, 4.2))
plt.plot(data_df["Year"], data_df["Total_National_Demand"], marker="o", linestyle="-", label="Actual (1986–2024)")
plt.plot(future_years, future_pred, marker="*", linestyle="--", label=f"Forecast ({forecast_start}–{forecast_end})")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Total_National_Demand (TWh)")
plt.legend()
st.pyplot(fig2, clear_figure=True)

# -------------------------
# Forecast table
# -------------------------
st.markdown("### Bảng dự báo")
out = pd.DataFrame({"Year": future_years, "Forecasted Value (TWh)": np.round(future_pred, 2)})
st.dataframe(out, use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(" Tải forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")
