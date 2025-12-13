import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from statsmodels.tsa.api import ExponentialSmoothing

# -------------------------
# Metrics
# -------------------------
def mae(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)

def compute_metrics(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE(%)": mape(y_true, y_pred)}

# -------------------------
# Model helpers
# -------------------------
def fit_linear(train_df: pd.DataFrame):
    model = LinearRegression()
    model.fit(train_df[["Year"]], train_df["Total_National_Demand"])
    return model

def predict_linear(model, years: np.ndarray):
    df = pd.DataFrame({"Year": years})
    return model.predict(df)

def fit_poly_with_tscv(train_df: pd.DataFrame, degrees=(2,3,4,5), n_splits=5):
    X = train_df[["Year"]].values
    y = train_df["Total_National_Demand"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best = None  # (avg_mae, degree, model)

    for d in degrees:
        fold_maes = []
        for tr_idx, va_idx in tscv.split(X):
            m = make_pipeline(PolynomialFeatures(degree=d, include_bias=False), LinearRegression())
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[va_idx])
            fold_maes.append(mae(y[va_idx], pred))
        avg = float(np.mean(fold_maes))
        if best is None or avg < best[0]:
            best = (avg, d)

    # fit final on full train
    _, best_degree = best
    final_model = make_pipeline(PolynomialFeatures(degree=best_degree, include_bias=False), LinearRegression())
    final_model.fit(X, y)

    return final_model, {"best_degree": int(best_degree), "cv_mae": float(best[0])}

def predict_poly(model, years: np.ndarray):
    X = years.reshape(-1, 1)
    return model.predict(X)

def fit_holt_additive(train_series: np.ndarray):
    # estimated init, additive trend, no damping
    model = ExponentialSmoothing(train_series, trend="add", damped_trend=False, initialization_method="estimated").fit()
    return model

def predict_holt(model, steps: int):
    return np.asarray(model.forecast(steps), dtype=float)

def fit_svr_rbf(train_df: pd.DataFrame, n_splits=5):
    X = train_df[["Year"]].values
    y = train_df["Total_National_Demand"].values.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(y).ravel()

    base = SVR(kernel="rbf")
    param_grid = {"C": [10, 100, 1000, 5000], "gamma": [0.1, 0.05, 0.01, 0.005]}
    tscv = TimeSeriesSplit(n_splits=n_splits)

    gs = GridSearchCV(base, param_grid=param_grid, scoring="neg_mean_absolute_error", cv=tscv)
    gs.fit(Xs, ys)

    best = gs.best_estimator_
    best.fit(Xs, ys)

    meta = {"best_params": gs.best_params_, "cv_mae": float(-gs.best_score_)}

    return (best, x_scaler, y_scaler), meta

def predict_svr(bundle, years: np.ndarray):
    model, x_scaler, y_scaler = bundle
    X = years.reshape(-1, 1)
    Xs = x_scaler.transform(X)
    pred_s = model.predict(Xs).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_s).ravel()
    return pred

# -------------------------
# Optional PSO-GM(1,1)
# -------------------------
def gm11_lambda_predict(x0: np.ndarray, lambda_val: float, n_forecast: int):
    """
    Returns fitted+forecasted series length len(x0)+n_forecast using GM(1,1) with background coefficient lambda.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    x1 = np.cumsum(x0)  # AGO

    # background series z1(k) = lambda*x1(k) + (1-lambda)*x1(k-1), k=2..n
    z1 = np.array([lambda_val * x1[k] + (1 - lambda_val) * x1[k-1] for k in range(1, n)], dtype=float)

    # Construct B and Y (least squares)
    B = np.column_stack((-z1, np.ones(n-1)))
    Y = x0[1:].reshape(-1, 1)

    # Solve [a,b]
    coef = np.linalg.lstsq(B, Y, rcond=None)[0].ravel()
    a, b = float(coef[0]), float(coef[1])

    # x1_hat(k+1) = (x0(1) - b/a)*exp(-a*k) + b/a
    def x1_hat(k):
        return (x0[0] - b / a) * np.exp(-a * k) + b / a

    x1_pred = np.array([x1_hat(k) for k in range(0, n + n_forecast)], dtype=float)
    # IAGO
    x0_pred = np.diff(x1_pred, prepend=x1_pred[0])
    return x0_pred

def fit_pso_gm11(train_series: np.ndarray, n_splits=5, iters=40, particles=25):
    """
    PSO to tune lambda in [0,1] by minimizing average MAE across time-series CV folds.
    Requires pyswarms. If not installed, raise ImportError.
    """
    try:
        import pyswarms as ps
    except Exception as e:
        raise ImportError("pyswarms is required for PSO-GM(1,1). Install it or disable this model.") from e

    x0 = np.asarray(train_series, dtype=float)

    # Build splits over indices for y
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def cost(lambdas):
        # lambdas shape: (n_particles, 1)
        vals = []
        for lam in lambdas.ravel():
            fold_maes = []
            for tr_idx, va_idx in tscv.split(x0):
                tr = x0[tr_idx]
                # forecast length = len(va)
                pred_full = gm11_lambda_predict(tr, float(lam), n_forecast=len(va_idx))
                pred = pred_full[-len(va_idx):]
                fold_maes.append(mae(x0[va_idx], pred))
            vals.append(np.mean(fold_maes))
        return np.array(vals)

    bounds = (np.array([0.0]), np.array([1.0]))
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=1, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(cost, iters=iters, verbose=False)
    best_lambda = float(best_pos[0])
    return best_lambda, {"cv_mae": float(best_cost)}

