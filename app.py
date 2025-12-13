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

uploaded = st.file_uploader("Upload dataset (.xlsx/.csv)", type=["xlsx", "csv"])

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    # bỏ cột trống + strip tên cột
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[1] < 2:
        raise ValueError("File phải có ít nhất 2 cột: (Year, Target).")

    # LẤY CỨNG THEO VỊ TRÍ: cột 1 = Year, cột 2 = Target
    year_col = df.columns[0]
    target_col = df.columns[1]

    out = df[[year_col, target_col]].copy()
    out.columns = ["Year", "Total_National_Demand"]  # chuẩn hoá tên nội bộ

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["Total_National_Demand"] = pd.to_numeric(out["Total_National_Demand"], errors="coerce")

    out = out.dropna().sort_values("Year")

    # Year phải là số nguyên (cho đẹp)
    out["Year"] = out["Year"].astype(int)

    return out

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

# -------------------------
# Metrics
# -------------------------
left_metrics, right_metrics = st.columns([0.6, 2.0])

with left_metrics:
    year_col, _ = st.columns([5, 2])
    with year_col:
        # with colA:
        model_name = st.selectbox(
            "Chọn mô hình",
            ["Holt (Additive)", "SVR (RBF)", "Polynomial Regression", "Linear Regression", "PSO-GM(1,1)"],
            index=0
        )
        forecast_start = st.number_input("Forecast start year",value=2025,min_value=1900,max_value=2100,step=1)
        forecast_end = st.number_input("Forecast end year",value=2030,min_value=1900,max_value=2100,step=1)

    if forecast_end < forecast_start:
        st.error("Forecast end year phải >= start year.")
        st.stop()

    future_years = np.arange(int(forecast_start), int(forecast_end) + 1)

    # with colC:
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
                pred_full_future = gm11_lambda_predict(data_df["Total_National_Demand"].to_numpy(), best_lambda_full,
                                                       n_forecast=len(future_years))
                future_pred = pred_full_future[-len(future_years):]
                meta["best_lambda"] = float(best_lambda)
            except Exception as e:
                st.error(str(e))
                st.stop()

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    st.markdown("### Metrics (Train)")
    st.write(f"- **MAE**: {train_metrics['MAE']:.4f}")
    st.write(f"- **RMSE**: {train_metrics['RMSE']:.4f}")
    st.write(f"- **MAPE(%)**: {train_metrics['MAPE(%)']:.4f}")

    st.markdown("### Metrics (Test)")
    st.write(f"- **MAE**: {test_metrics['MAE']:.4f}")
    st.write(f"- **RMSE**: {test_metrics['RMSE']:.4f}")
    st.write(f"- **MAPE(%)**: {test_metrics['MAPE(%)']:.4f}")

    if meta:
        st.markdown("### Thông tin thêm")
        for k, v in meta.items():
            st.write(f"- **{k}**: {v}")

with right_metrics:
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
    st.empty()

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(" Tải forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")