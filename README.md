# Vietnam Energy Demand Forecasting (AI Web App)

This project is developed for the course **AI Data Analysis (Chuyên đề 1)**.

## Objective
Analyze and forecast Vietnam's national electricity demand.

## Models Used
- Holt’s Exponential Smoothing (Additive)
- Support Vector Regression (RBF)
- Linear Regression
- Polynomial Regression
- (Optional) GM(1,1) + PSO (enabled if `pyswarms` is installed)

## Dataset
- Historical electricity demand data (1986–2024)
- Target: Total National Demand (TWh)

## Web Application
The Streamlit web app allows users to:
- Upload data (Excel/CSV)
- Select forecasting models
- Evaluate performance (MAE, RMSE, MAPE)
- Forecast electricity demand for 2025–2030

## Deployment
The application is deployed using **Streamlit Community Cloud**.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py



## Features
- Upload your dataset (Excel or CSV) with columns: `Year`, `Total_National_Demand` (TWh)
- Train/test split: 1986–2016 train, 2017–2024 test (same as the report/notebook)
- Models:
  - Holt’s Exponential Smoothing (Additive trend)
  - SVR (RBF) + scaling
  - Linear Regression
  - Polynomial Regression (degree chosen by time-series CV)
  - (Optional) GM(1,1) + PSO (enabled if `pyswarms` is installed)

- Metrics: MAE, RMSE, MAPE
- Plots: actual vs predicted (train/test) + 2025–2030 forecast

## Quickstart
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
#source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Data format
Example columns:

| Year | Total_National_Demand |
|------|------------------------|
| 1986 |  ...                   |
| ...  |  ...                   |
| 2024 |  ...                   |

If your column names differ, rename them in your file before uploading.
