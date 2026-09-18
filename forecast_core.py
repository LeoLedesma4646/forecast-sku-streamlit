from __future__ import annotations

from pathlib import Path
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# Configuración metodológica
# -----------------------------
PERIOD = 52                 # ciclo anual para datos semanales
MIN_WEEKS = 104             # 2 ciclos completos de 52 semanas
TOP_N = 10
FOURIER_K = 2
BACKTEST_FOLDS = 3
BACKTEST_HORIZON = 4
TUNING_HORIZON = 4

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
DHR_ORDERS = ((1, 0, 0), (0, 0, 1), (1, 0, 1))

# SARIMAX genuinamente estacional y parsimonioso. Se compara un término
# estacional AR(1) frente a uno MA(1), ambos a 52 semanas.
SEASONAL_SARIMAX_CANDIDATES = (
    ((1, 0, 0), (1, 0, 0, PERIOD)),
    ((1, 0, 0), (0, 0, 1, PERIOD)),
)

MODEL_NAIVE = "Seasonal Naive"
MODEL_RIDGE = "Fourier + Ridge"
MODEL_SARIMAX = "SARIMAX estacional"
MODEL_DHR = "Fourier + ARMA (DHR)"
MODEL_NAMES = (MODEL_NAIVE, MODEL_RIDGE, MODEL_SARIMAX, MODEL_DHR)

EXOG_BASE = [
    "promotion_flag",
    "price_unit",
    "is_summer",
    "is_winter",
    "is_holiday_week",
]


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="weekly_df_final_for_modeling")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["price_unit"] = pd.to_numeric(df["price_unit"], errors="coerce")
    for c in ["promotion_flag", "is_summer", "is_winter", "is_holiday_week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def common_span(df_sku: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    spans = df_sku.groupby(["channel", "region"], observed=True)["week"].agg(["min", "max"])
    return spans["min"].max(), spans["max"].min()


def sku_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku, g in df.dropna(subset=["sku", "week"]).groupby("sku", observed=True):
        start, end = common_span(g)
        common_weeks = (
            len(pd.date_range(start, end, freq="W-MON"))
            if pd.notna(start) and pd.notna(end) and start <= end else 0
        )
        revenue = (g["units_sold"] * g["price_unit"]).sum(min_count=1)
        units = g["units_sold"].sum(min_count=1)
        rows.append({
            "sku": str(sku),
            "common_weeks": int(common_weeks),
            "seasonal_cycles": float(common_weeks / PERIOD),
            "admissible": bool(common_weeks >= MIN_WEEKS),
            "total_revenue": float(revenue) if pd.notna(revenue) else np.nan,
            "total_units": float(units) if pd.notna(units) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["common_weeks", "total_revenue"], ascending=[False, False]
    ).reset_index(drop=True)


def select_top_skus(summary: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    return (
        summary[summary["admissible"]]
        .sort_values(["total_revenue", "total_units"], ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def aggregate_sku(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Agrega canales/regiones a una única serie semanal por SKU usando el intervalo común."""
    g = df[df["sku"].astype(str).eq(str(sku))].copy()
    if g.empty:
        raise ValueError(f"SKU no encontrado: {sku}")
    start, end = common_span(g)
    g = g[g["week"].between(start, end)].copy()

    out = (
        g.groupby("week", observed=True)
        .agg(
            units_sold=("units_sold", lambda s: s.sum(min_count=1)),
            price_unit=("price_unit", "mean"),
            promotion_flag=("promotion_flag", "mean"),
            is_summer=("is_summer", "max"),
            is_winter=("is_winter", "max"),
            is_holiday_week=("is_holiday_week", "max"),
        )
        .reset_index()
        .sort_values("week")
        .reset_index(drop=True)
    )
    if out["units_sold"].isna().any():
        raise ValueError(
            f"{sku} contiene ventas no numéricas dentro del intervalo común. "
            "Corrige la fuente antes de entrenar este SKU."
        )
    return out


# -----------------------------
# Diseño de predictores
# -----------------------------
def add_fourier_features(frame: pd.DataFrame, start_week, k: int = FOURIER_K) -> pd.DataFrame:
    """Tendencia + Fourier anual + variables exógenas. Solo para Ridge y DHR."""
    start_week = pd.Timestamp(start_week)
    t = ((pd.DatetimeIndex(frame["week"]) - start_week).days / 7).astype(float)
    X = pd.DataFrame(index=range(len(frame)))
    X["t"] = t
    X["t2"] = t ** 2
    for harmonic in range(1, k + 1):
        X[f"sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * t / PERIOD)
        X[f"cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * t / PERIOD)
    for c in EXOG_BASE:
        X[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0).to_numpy(float)
    return X.astype(float)


def add_sarimax_exog(frame: pd.DataFrame) -> pd.DataFrame:
    """Exógenas del SARIMAX estacional. No contiene Fourier ni t/t²."""
    X = pd.DataFrame(index=range(len(frame)))
    for c in EXOG_BASE:
        X[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0).to_numpy(float)
    return X.astype(float)


# -----------------------------
# Métricas
# -----------------------------
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    den = np.sum(np.abs(y_true))
    return float(100 * np.sum(np.abs(y_true - y_pred)) / den) if den else np.nan


def _score(y_true, y_pred) -> tuple[float, float]:
    return wape(y_true, y_pred), mae(y_true, y_pred)


# -----------------------------
# Cuatro modelos candidatos
# -----------------------------
def seasonal_naive(train: pd.DataFrame, horizon: int) -> np.ndarray:
    y = train["units_sold"].to_numpy(float)
    if len(y) < PERIOD:
        raise ValueError("Seasonal Naive requiere al menos 52 semanas.")
    season = y[-PERIOD:]
    return np.resize(season, horizon).astype(float)


def fit_fourier_ridge(train: pd.DataFrame, k: int = FOURIER_K, alpha: float = 10.0):
    X = add_fourier_features(train, train["week"].iloc[0], k=k)
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(X, train["units_sold"].to_numpy(float))
    return model


def fit_seasonal_sarimax(
    train: pd.DataFrame,
    order=(1, 0, 0),
    seasonal_order=(1, 0, 0, PERIOD),
):
    """SARIMAX estacional genuino: sin Fourier; estacionalidad mediante rezagos de 52 semanas."""
    X = add_sarimax_exog(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train["units_sold"].astype(float),
            exog=X,
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            trend="ct",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=80, method="powell")
    return result


def fit_dhr_arma(train: pd.DataFrame, k: int = FOURIER_K, order=(1, 0, 1)):
    """Regresión armónica dinámica: Fourier + exógenas + errores ARMA no estacionales."""
    X = add_fourier_features(train, train["week"].iloc[0], k=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train["units_sold"].astype(float),
            exog=X,
            order=tuple(order),
            seasonal_order=(0, 0, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=70)
    return result


def _is_converged(result) -> bool:
    try:
        return bool(result.mle_retvals.get("converged", True))
    except Exception:
        return True


def _forecast_sarimax_result(result, horizon: int, exog: pd.DataFrame) -> np.ndarray:
    return np.maximum(
        np.asarray(result.get_forecast(horizon, exog=exog).predicted_mean, dtype=float), 0
    )


# -----------------------------
# Calibración SIN mirar el test final
# -----------------------------
def select_hyperparameters(
    series: pd.DataFrame,
    folds: int = BACKTEST_FOLDS,
    horizon: int = BACKTEST_HORIZON,
    tuning_horizon: int = TUNING_HORIZON,
    k: int = FOURIER_K,
) -> dict:
    """
    Selecciona parámetros usando un bloque temporal inmediatamente anterior al backtest externo.
    Las últimas folds*horizon semanas quedan completamente fuera de esta calibración.
    """
    outer_test_start = len(series) - folds * horizon
    development = series.iloc[:outer_test_start].copy()
    if len(development) <= tuning_horizon + PERIOD:
        raise ValueError("Historia insuficiente para separar calibración y backtest.")

    tune_train = development.iloc[:-tuning_horizon].copy()
    tune_val = development.iloc[-tuning_horizon:].copy()
    y_val = tune_val["units_sold"].to_numpy(float)

    # Ridge alpha
    ridge_rows = []
    for alpha in RIDGE_ALPHAS:
        model = fit_fourier_ridge(tune_train, k=k, alpha=alpha)
        Xv = add_fourier_features(tune_val, tune_train["week"].iloc[0], k=k)
        pred = np.maximum(np.asarray(model.predict(Xv), float), 0)
        sw, sm = _score(y_val, pred)
        ridge_rows.append((sw, sm, float(alpha)))
    ridge_rows.sort(key=lambda x: (x[0], x[1]))
    ridge_alpha = ridge_rows[0][2]

    # DHR: orden ARMA del error
    dhr_rows = []
    Xv_fourier = add_fourier_features(tune_val, tune_train["week"].iloc[0], k=k)
    for order in DHR_ORDERS:
        try:
            result = fit_dhr_arma(tune_train, k=k, order=order)
            pred = _forecast_sarimax_result(result, tuning_horizon, Xv_fourier)
            sw, sm = _score(y_val, pred)
            dhr_rows.append((sw, sm, tuple(order), _is_converged(result)))
        except Exception:
            continue
    converged_dhr = [r for r in dhr_rows if r[3]] or dhr_rows
    if not converged_dhr:
        dhr_order = (1, 0, 1)
    else:
        converged_dhr.sort(key=lambda x: (x[0], x[1]))
        dhr_order = converged_dhr[0][2]

    # SARIMAX estacional: dos estructuras parsimoniosas a s=52
    sarimax_rows = []
    Xv_sarimax = add_sarimax_exog(tune_val)
    for order, seasonal_order in SEASONAL_SARIMAX_CANDIDATES:
        try:
            result = fit_seasonal_sarimax(
                tune_train, order=order, seasonal_order=seasonal_order
            )
            pred = _forecast_sarimax_result(result, tuning_horizon, Xv_sarimax)
            sw, sm = _score(y_val, pred)
            sarimax_rows.append(
                (sw, sm, tuple(order), tuple(seasonal_order), _is_converged(result))
            )
        except Exception:
            continue
    converged_sarimax = [r for r in sarimax_rows if r[4]] or sarimax_rows
    if not converged_sarimax:
        sarimax_order = (1, 0, 0)
        sarimax_seasonal_order = (1, 0, 0, PERIOD)
    else:
        converged_sarimax.sort(key=lambda x: (x[0], x[1]))
        sarimax_order = converged_sarimax[0][2]
        sarimax_seasonal_order = converged_sarimax[0][3]

    return {
        "fourier_k": int(k),
        "ridge_alpha": float(ridge_alpha),
        "dhr_order": tuple(dhr_order),
        "sarimax_order": tuple(sarimax_order),
        "sarimax_seasonal_order": tuple(sarimax_seasonal_order),
        "tuning_train_weeks": int(len(tune_train)),
        "tuning_validation_weeks": int(len(tune_val)),
        "outer_test_weeks_reserved": int(folds * horizon),
    }


# -----------------------------
# Backtest externo común a los 4 modelos
# -----------------------------
def rolling_backtest(
    series: pd.DataFrame,
    folds: int = BACKTEST_FOLDS,
    horizon: int = BACKTEST_HORIZON,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(series) < MIN_WEEKS:
        raise ValueError("La serie no cumple el mínimo de 104 semanas.")
    if folds * horizon >= len(series) - PERIOD:
        raise ValueError("Configuración de backtest deja muy poca historia.")

    params = params or select_hyperparameters(series, folds=folds, horizon=horizon)
    k = int(params["fourier_k"])
    alpha = float(params["ridge_alpha"])
    dhr_order = tuple(params["dhr_order"])
    sarimax_order = tuple(params["sarimax_order"])
    sarimax_seasonal_order = tuple(params["sarimax_seasonal_order"])

    rows = []
    first_test = len(series) - folds * horizon

    for fold in range(folds):
        test_start = first_test + fold * horizon
        test_end = test_start + horizon
        train = series.iloc[:test_start].copy()
        test = series.iloc[test_start:test_end].copy()

        preds = {MODEL_NAIVE: seasonal_naive(train, horizon)}

        ridge = fit_fourier_ridge(train, k=k, alpha=alpha)
        X_test_fourier = add_fourier_features(test, train["week"].iloc[0], k=k)
        preds[MODEL_RIDGE] = np.maximum(np.asarray(ridge.predict(X_test_fourier)), 0)

        seasonal = fit_seasonal_sarimax(
            train, order=sarimax_order, seasonal_order=sarimax_seasonal_order
        )
        X_test_sarimax = add_sarimax_exog(test)
        preds[MODEL_SARIMAX] = _forecast_sarimax_result(
            seasonal, horizon, X_test_sarimax
        )

        dhr = fit_dhr_arma(train, k=k, order=dhr_order)
        preds[MODEL_DHR] = _forecast_sarimax_result(
            dhr, horizon, X_test_fourier
        )

        for model_name, pred in preds.items():
            for week, actual, forecast in zip(test["week"], test["units_sold"], pred):
                rows.append({
                    "fold": fold + 1,
                    "week": week,
                    "model": model_name,
                    "actual": float(actual),
                    "prediction": float(forecast),
                    "residual": float(actual - forecast),
                })

    predictions = pd.DataFrame(rows)
    metrics = (
        predictions.groupby("model", observed=True)
        .apply(
            lambda g: pd.Series({
                "MAE": mae(g["actual"], g["prediction"]),
                "WAPE": wape(g["actual"], g["prediction"]),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values(["WAPE", "MAE"])
        .reset_index(drop=True)
    )
    metrics["rank"] = np.arange(1, len(metrics) + 1)
    return predictions, metrics


# -----------------------------
# Exploración
# -----------------------------
def correlation_table(series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in EXOG_BASE:
        d = series[["units_sold", c]].dropna()
        if len(d) < 5 or d[c].nunique() < 2:
            rows.append({"variable": c, "spearman_rho": np.nan, "p_value": np.nan})
            continue
        rho, p = stats.spearmanr(d[c], d["units_sold"])
        rows.append({"variable": c, "spearman_rho": float(rho), "p_value": float(p)})
    out = pd.DataFrame(rows)
    out["abs_rho"] = out["spearman_rho"].abs()
    return out.sort_values("abs_rho", ascending=False).reset_index(drop=True)


def decompose_stl(series: pd.DataFrame, period: int = PERIOD):
    if len(series) < 2 * period:
        raise ValueError("STL anual requiere al menos 104 semanas.")
    y = series.set_index("week")["units_sold"].astype(float)
    return STL(y, period=period, robust=True).fit()


def infer_season_months(df: pd.DataFrame) -> tuple[set[int], set[int]]:
    summer = set(
        df.groupby(df["week"].dt.month)["is_summer"].mean()
        .loc[lambda s: s >= 0.5].index.astype(int)
    )
    winter = set(
        df.groupby(df["week"].dt.month)["is_winter"].mean()
        .loc[lambda s: s >= 0.5].index.astype(int)
    )
    return summer, winter


def future_scenario(
    history: pd.DataFrame,
    horizon: int,
    price_change_pct: float = 0.0,
    promotion_share: float | None = None,
    holiday: bool = False,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    last_week = pd.Timestamp(history["week"].iloc[-1])
    weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    base_price = float(history["price_unit"].tail(4).mean())
    base_promo = float(history["promotion_flag"].tail(4).mean())
    promo = base_promo if promotion_share is None else float(promotion_share)

    if source_df is not None:
        summer_months, winter_months = infer_season_months(source_df)
    else:
        summer_months, winter_months = set(), set()

    future = pd.DataFrame({"week": weeks})
    future["price_unit"] = base_price * (1 + price_change_pct / 100)
    future["promotion_flag"] = np.clip(promo, 0, 1)
    future["is_summer"] = future["week"].dt.month.isin(summer_months).astype(int)
    future["is_winter"] = future["week"].dt.month.isin(winter_months).astype(int)
    future["is_holiday_week"] = int(bool(holiday))
    return future


# -----------------------------
# Persistencia de modelos
# -----------------------------
def ridge_coefficients(model, feature_names: list[str], sku: str) -> pd.DataFrame:
    ridge = model.named_steps["ridge"]
    return pd.DataFrame({
        "sku": sku,
        "feature": feature_names,
        "standardized_coefficient": np.asarray(ridge.coef_, dtype=float),
    }).sort_values("standardized_coefficient", key=lambda s: s.abs(), ascending=False)


def fit_and_save_sku(
    series: pd.DataFrame,
    sku: str,
    model_dir: str | Path,
    metrics: pd.DataFrame | None = None,
    params: dict | None = None,
) -> dict:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    params = params or select_hyperparameters(series)

    k = int(params["fourier_k"])
    ridge = fit_fourier_ridge(series, k=k, alpha=float(params["ridge_alpha"]))
    seasonal = fit_seasonal_sarimax(
        series,
        order=tuple(params["sarimax_order"]),
        seasonal_order=tuple(params["sarimax_seasonal_order"]),
    )
    dhr = fit_dhr_arma(series, k=k, order=tuple(params["dhr_order"]))

    with open(model_dir / f"{sku}_fourier_ridge.pkl", "wb") as f:
        pickle.dump(ridge, f)
    with open(model_dir / f"{sku}_seasonal_sarimax.pkl", "wb") as f:
        pickle.dump(seasonal, f)
    with open(model_dir / f"{sku}_dhr_arma.pkl", "wb") as f:
        pickle.dump(dhr, f)

    meta = {
        "sku": sku,
        "start_week": str(pd.Timestamp(series["week"].iloc[0]).date()),
        "end_week": str(pd.Timestamp(series["week"].iloc[-1]).date()),
        "n_weeks": int(len(series)),
        "seasonal_period": PERIOD,
        "seasonal_cycles": float(len(series) / PERIOD),
        "fourier_k": k,
        "ridge_alpha": float(params["ridge_alpha"]),
        "dhr_order": list(params["dhr_order"]),
        "sarimax_order": list(params["sarimax_order"]),
        "sarimax_seasonal_order": list(params["sarimax_seasonal_order"]),
        "seasonal_sarimax_converged": _is_converged(seasonal),
        "dhr_converged": _is_converged(dhr),
    }
    if metrics is not None and not metrics.empty:
        winner = metrics.sort_values(["WAPE", "MAE"]).iloc[0]
        meta["winner"] = str(winner["model"])
        meta["winner_WAPE"] = float(winner["WAPE"])
        meta["winner_MAE"] = float(winner["MAE"])

    (model_dir / f"{sku}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def load_saved_models(sku: str, model_dir: str | Path):
    model_dir = Path(model_dir)
    with open(model_dir / f"{sku}_fourier_ridge.pkl", "rb") as f:
        ridge = pickle.load(f)
    with open(model_dir / f"{sku}_seasonal_sarimax.pkl", "rb") as f:
        seasonal = pickle.load(f)
    with open(model_dir / f"{sku}_dhr_arma.pkl", "rb") as f:
        dhr = pickle.load(f)
    meta = json.loads((model_dir / f"{sku}_meta.json").read_text(encoding="utf-8"))
    return {"ridge": ridge, "seasonal_sarimax": seasonal, "dhr": dhr, "meta": meta}


def forecast_from_saved(
    history: pd.DataFrame,
    future: pd.DataFrame,
    model_name: str,
    models: dict,
) -> np.ndarray:
    horizon = len(future)
    if model_name == MODEL_NAIVE:
        return np.maximum(seasonal_naive(history, horizon), 0)

    meta = models["meta"]
    k = int(meta.get("fourier_k", FOURIER_K))

    if model_name == MODEL_RIDGE:
        X_future = add_fourier_features(future, history["week"].iloc[0], k=k)
        return np.maximum(np.asarray(models["ridge"].predict(X_future)), 0)

    if model_name == MODEL_SARIMAX:
        X_future = add_sarimax_exog(future)
        return _forecast_sarimax_result(models["seasonal_sarimax"], horizon, X_future)

    if model_name == MODEL_DHR:
        X_future = add_fourier_features(future, history["week"].iloc[0], k=k)
        return _forecast_sarimax_result(models["dhr"], horizon, X_future)

    raise ValueError(f"Modelo desconocido: {model_name}")
